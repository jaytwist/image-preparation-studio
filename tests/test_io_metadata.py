"""Decode/encode, colour management, bit depth, alpha, orientation, metadata transfer + audit."""
import io
import threading

import numpy as np
import pytest
from PIL import Image

from ips.core import imageio, metadata
from ips.core.hashing import sha256_file
from ips.core.outputs import ExportProfile, write_output


def _roundtrip(src_path, prof, rgb=None, alpha=None, tmp_path=None, scale=1.0):
    im = imageio.load_image(src_path)
    md = metadata.read_metadata(src_path)
    rgb = im.rgb if rgb is None else rgb
    alpha = im.alpha if alpha is None else alpha
    dest = tmp_path / f"out{imageio.SUPPORTED_EXTS and ''}{'.tif' if prof.fmt == 'TIFF' else '.' + prof.fmt.lower()}"
    final, audit, dims, data = write_output(rgb, alpha, im.icc_profile, prof, md, dest, source_bits=im.bit_depth,
                                            source_sha="ab" * 32, recipe_hash="cd" * 8, recipe_summary="test",
                                            scale=scale, generative=False, edited=False, png_text=md.png_text)
    return im, md, final, audit, dims


def test_orientation_applied_and_reset(corpus, tmp_path):
    p = corpus / "metadata/portrait_6.jpg"
    raw = Image.open(p)
    w0, h0 = raw.size
    im = imageio.load_image(p)
    assert (im.width, im.height) == (h0, w0), "Orientation 6 must rotate pixels"
    _, _, final, audit, _ = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="TIFF"), tmp_path=tmp_path)
    md2 = metadata.read_metadata(final)
    assert md2.exif.get("Exif.Image.Orientation") == "1"
    assert audit["updated"]["Exif.Image.Orientation"]["from"] == "6"
    assert md2.exif["Exif.Photo.PixelXDimension"] == str(im.width)


def test_full_metadata_transfer_and_gps_strip(corpus, tmp_path):
    p = corpus / "metadata/full_metadata.jpg"
    src = metadata.read_metadata(p)
    assert any(k.startswith("Exif.GPSInfo.GPSLatitude") for k in src.exif)
    _, _, final, audit, _ = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="TIFF"), tmp_path=tmp_path)
    out = metadata.read_metadata(final)
    assert audit["verified"], audit.get("verify_mismatches")
    assert out.exif["Exif.Photo.DateTimeOriginal"] == src.exif["Exif.Photo.DateTimeOriginal"]
    assert "Exif.GPSInfo.GPSLatitude" in out.exif, "archive keeps GPS by default"
    assert out.iptc.get("Iptc.Application2.Keywords") == src.iptc["Iptc.Application2.Keywords"]
    assert out.xmp.get("Xmp.dc.rights") is not None
    assert "Xmp.crs.Exposure2012" not in out.xmp, "Camera Raw develop settings must not be carried"
    assert "Xmp.crs.Exposure2012" in audit["dropped"]
    assert out.xmp["Xmp.ips.SourceSHA256"] == "ab" * 32
    # share export with location removal
    d2 = tmp_path / "share"
    d2.mkdir()
    _, _, f2, a2, _ = _roundtrip(p, ExportProfile("share", "share_export", fmt="JPEG", bits="8", to_srgb=True,
                                                  strip_location=True), tmp_path=d2)
    o2 = metadata.read_metadata(f2)
    assert not any(k.startswith("Exif.GPSInfo") for k in o2.exif)
    assert "Iptc.Application2.City" not in o2.iptc and "Xmp.photoshop.City" not in o2.xmp
    assert a2["location_leaks"] == [] and a2["verified"]
    assert o2.exif["Exif.Photo.DateTimeOriginal"] == src.exif["Exif.Photo.DateTimeOriginal"]


def test_makernote_dropped_and_disclosed(corpus, tmp_path):
    p = corpus / "metadata/Canon_40D.jpg"
    _, src, final, audit, _ = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="TIFF"), tmp_path=tmp_path)
    vendor = [k for k in src.exif if "MakerNote" in k or k.startswith(("Exif.Canon", "Exif.Nikon", "Exif.Sony"))]
    if vendor:
        assert all(k in audit["dropped"] for k in vendor)
        assert any("MakerNote" in n for n in audit["notes"])
    # thumbnails (IFD1) always describe the unprocessed pixels
    assert all(k in audit["dropped"] for k in src.exif if k.startswith("Exif.Thumbnail."))
    out = metadata.read_metadata(final)
    assert out.exif.get("Exif.Image.Model") == src.exif.get("Exif.Image.Model")


def test_malformed_exif_does_not_crash(corpus, tmp_path):
    p = corpus / "metadata/malformed_exif.jpg"
    im = imageio.load_image(p)
    assert im.width > 0
    _, md, final, audit, _ = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="PNG"), tmp_path=tmp_path)
    assert final.is_file()


@pytest.mark.parametrize("fname,icc_word", [("metadata/adobergb.jpg", "a98"),
                                            ("metadata/prophoto_16bit.tif", "romm")])
def test_wide_gamut_icc_preserved_in_archive_and_converted_for_share(corpus, tmp_path, fname, icc_word):
    p = corpus / fname
    im = imageio.load_image(p)
    assert im.icc_profile and icc_word in imageio.icc_description(im.icc_profile).lower()
    assert not imageio.icc_is_srgb(im.icc_profile), "wide-gamut profile must not be mistaken for sRGB"
    _, _, f1, a1, (_, _, bits) = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="TIFF"),
                                            tmp_path=tmp_path)
    o1 = imageio.load_image(f1)
    assert icc_word in imageio.icc_description(o1.icc_profile).lower(), "archive keeps source ICC"
    assert bits == im.bit_depth
    np.testing.assert_allclose(o1.rgb, im.rgb, atol=1 / 255 if bits == 8 else 2 / 65535)
    d = tmp_path / "s"
    d.mkdir()
    _, _, f2, a2, _ = _roundtrip(p, ExportProfile("share", "share_export", fmt="JPEG", bits="8", to_srgb=True),
                                 tmp_path=d)
    o2 = imageio.load_image(f2)
    assert imageio.icc_is_srgb(o2.icc_profile)
    assert any("sRGB" in n for n in a2["profile_notes"])
    md = metadata.read_metadata(f2)
    assert md.exif.get("Exif.Photo.ColorSpace") == "1"


def test_16bit_alpha_png_preserved(corpus, tmp_path):
    p = corpus / "metadata/tree_alpha_16bit.png"
    im = imageio.load_image(p)
    assert im.bit_depth == 16 and im.alpha is not None
    _, _, f, audit, (_, _, bits) = _roundtrip(p, ExportProfile("cutout", "cutout", fmt="PNG"), tmp_path=tmp_path)
    o = imageio.load_image(f)
    assert bits == 16 and o.bit_depth == 16 and o.alpha is not None
    np.testing.assert_allclose(o.alpha, im.alpha, atol=2 / 65535)
    np.testing.assert_allclose(o.rgb, im.rgb, atol=2 / 65535)


def test_16bit_tiff_with_p3_icc(corpus, tmp_path):
    p = corpus / "dslr/mcm1_16bit_p3_noisy.tif"
    im = imageio.load_image(p)
    assert im.bit_depth == 16 and "p3" in imageio.icc_description(im.icc_profile).lower()
    assert len(np.unique((im.rgb[..., 0] * 65535).astype(np.uint16))) > 256, "genuine 16-bit samples"


def test_heic_decode_and_metadata(corpus, tmp_path):
    p = corpus / "metadata/astronaut.heic"
    if not p.is_file():
        pytest.skip("HEIC fixture unavailable")
    im = imageio.load_image(p)
    assert (im.width, im.height) == (512, 512)
    md = metadata.read_metadata(p)
    s = metadata.summarize(md, im.width, im.height)
    assert s["camera_make"] == "Apple" or md.reader == "pillow"
    _, _, f, audit, _ = _roundtrip(p, ExportProfile("archive", "restored_master", fmt="TIFF"), tmp_path=tmp_path)
    out = metadata.read_metadata(f)
    assert out.exif.get("Exif.Photo.DateTimeOriginal") == "2024:06:01 09:00:00"


def test_raw_nef_developed_not_written_as_raw(corpus, tmp_path):
    p = corpus / "dslr/iss030e122639.NEF"
    if not p.is_file():
        pytest.skip("NEF fixture unavailable")
    im = imageio.load_image(p)
    assert im.raw_developed and im.bit_depth == 16 and im.width > 1000
    assert any("not RAW" in n or "RAW files" in n for n in im.decode_notes)
    md = metadata.read_metadata(p)
    assert "NIKON" in str(md.exif.get("Exif.Image.Make", "")).upper()


def test_png_generation_metadata_carried_in_png(corpus, tmp_path):
    p = corpus / "ai/scene_comfy.png"
    _, md, f, audit, _ = _roundtrip(p, ExportProfile("ai_ready", "ai_ready", fmt="PNG", bits="8", to_srgb=True),
                                   tmp_path=tmp_path)
    assert "prompt" in md.png_text
    with Image.open(f) as o:
        assert "prompt" in o.info and "workflow" in o.info
    d = tmp_path / "j"
    d.mkdir()
    _, _, f2, a2, _ = _roundtrip(p, ExportProfile("share", "share_export", fmt="JPEG", bits="8"), tmp_path=d)
    assert "PNG:generation_text" in a2["dropped"]


def test_no_clobber_versioning(tmp_path):
    dest = tmp_path / "a" / "x.png"
    p1 = imageio.write_bytes_no_clobber(b"one", dest)
    p2 = imageio.write_bytes_no_clobber(b"two", dest)
    p3 = imageio.write_bytes_no_clobber(b"three", dest)
    assert p1 == dest and p2.name == "x_v002.png" and p3.name == "x_v003.png"
    assert dest.read_bytes() == b"one" and p2.read_bytes() == b"two"
    assert not list(dest.parent.glob("*.part"))


def test_metadata_writes_serialised_under_concurrency(corpus, tmp_path):
    """Concurrent exports must not corrupt metadata (pyexiv2 is not thread-safe -> global lock)."""
    p = corpus / "metadata/full_metadata.jpg"
    im = imageio.load_image(p)
    md = metadata.read_metadata(p)
    errs, audits = [], []

    def work(i):
        try:
            d = tmp_path / f"t{i}"
            prof = ExportProfile("share", "share_export", fmt="JPEG", bits="8", strip_location=(i % 2 == 0))
            f, a, _, _ = write_output(im.rgb, None, im.icc_profile, prof, md, d / "o.jpg", source_bits=8,
                                      source_sha=f"{i:064d}", recipe_hash="x" * 16, recipe_summary=f"r{i}",
                                      scale=1, generative=False, edited=False)
            out = metadata.read_metadata(f)
            assert out.xmp["Xmp.ips.SourceSHA256"] == f"{i:064d}"
            assert ("Exif.GPSInfo.GPSLatitude" in out.exif) == (i % 2 == 1)
            audits.append(a["verified"])
        except Exception as e:  # pragma: no cover
            errs.append(repr(e))
    th = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs, errs
    assert all(audits) and len(audits) == 16


def test_source_never_modified_by_export(corpus, tmp_path, studio_nomodels):
    p = corpus / "metadata/full_metadata.jpg"
    before = sha256_file(p)
    mtime = p.stat().st_mtime_ns
    o = studio_nomodels.prepare(p, profiles=("archive", "share", "ai_ready"))
    assert sha256_file(p) == before and p.stat().st_mtime_ns == mtime
    assert len(o.outputs) == 3
    assert all(x.path.is_file() for x in o.outputs)
