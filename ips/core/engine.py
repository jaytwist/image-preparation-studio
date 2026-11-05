"""Studio engine: the single service used by the UI, CLI, batch queue and evaluation harness."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..settings import Settings
from . import classify, diagnostics, imageio, metadata
from .db import DB
from .hashing import dhash, sha256_bytes, sha256_file
from .outputs import ExportProfile, output_path, profiles_from_settings, write_output
from .perf import FastProfile
from .pipeline import Pipeline, ProcessResult
from .recipes import Recipe
from .router import RouteRequest, route

log = logging.getLogger("ips.engine")


@dataclass
class SourceRecord:
    path: Path
    sha256: str
    image: imageio.LoadedImage
    md: metadata.Metadata
    summary: dict[str, Any]
    diag: diagnostics.Diagnostics
    assessment: classify.Assessment
    dhash: int
    source_id: int | None = None
    rel_dir: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputRecord:
    id: int | None
    kind: str
    profile: str
    path: Path
    sha256: str
    width: int
    height: int
    bits: int
    audit: dict[str, Any]
    reused_master: bool = False


@dataclass
class JobOutcome:
    source: SourceRecord
    recipe: Recipe
    auto_recipe: Recipe
    result: ProcessResult | None
    outputs: list[OutputRecord]
    timings_ms: dict[str, float]
    review_flags: list[str]
    state: str                     # done | review | needs_override | failed
    error: str | None = None


class Studio:
    def __init__(self, settings: Settings | None = None, db: DB | None = None, *, load_models: bool = True):
        self.settings = settings or Settings.load()
        self.db = db or DB()
        self.pipeline = Pipeline()
        self.profile = FastProfile()
        self._models_ok: bool | None = None if load_models else False
        self._lock = threading.RLock()
        self._dev: Any = None

    # ------------------------------------------------------------------ device
    @property
    def device(self):
        if self._dev is None:
            try:
                from ..backends.direct import device_info
                self._dev = device_info(self.settings.device)
            except Exception:
                from ..backends.direct import DeviceInfo
                self._dev = DeviceInfo("cpu", "cpu (torch unavailable)")
        return self._dev

    @property
    def models_ok(self) -> bool:
        if self._models_ok is None:
            from ..backends.direct import torch_available
            self._models_ok = torch_available()
            if self._models_ok:
                from ..backends.direct import get_backend
                get_backend(device=self.settings.device, tile=self.settings.tile or None,
                            max_cached=self.settings.max_cached_models)
        return bool(self._models_ok)

    def face_detector(self):
        try:
            from .identity import FaceAnalyzer
            fa = FaceAnalyzer.get()
            return fa.detect, fa.name
        except Exception as e:
            log.warning("face analyzer unavailable: %s", e)
            return None, "none"

    # ------------------------------------------------------------------ inspect
    def inspect(self, path: str | os.PathLike, rel_dir: str = "", record: bool = True) -> SourceRecord:
        p = Path(path)
        t0 = time.perf_counter()
        data = p.read_bytes()
        sha = sha256_bytes(data)
        t1 = time.perf_counter()
        md = metadata.read_metadata(p, data)
        t2 = time.perf_counter()
        orient = None
        try:
            o = md.exif.get("Exif.Image.Orientation")
            orient = int(str(o).split()[0]) if o is not None else None
        except ValueError:
            orient = None
        img = imageio.load_image(p, orientation=orient)
        if not md.png_text and img.png_text:
            md.png_text = img.png_text
        if md.icc is None and img.icc_profile:
            md.icc = img.icc_profile
        if img.icc_profile is None and md.icc:
            img.icc_profile = md.icc
        t3 = time.perf_counter()
        summary = metadata.summarize(md, img.width, img.height)
        load_ms: dict[str, float] = {}
        from .identity import FaceAnalyzer
        if FaceAnalyzer._instance is None:
            tl = time.perf_counter()
            self.face_detector()
            load_ms["face_analyzer"] = (time.perf_counter() - tl) * 1e3
        det, det_name = self.face_detector()
        dg = diagnostics.diagnose(img.rgb, bit_depth=img.bit_depth, has_alpha=img.alpha is not None,
                                  face_detector=det, face_detector_name=det_name)
        if self.models_ok and (dg.features.get("blockiness", 0) > classify.THRESHOLDS["block_mild"] or
                               (img.jpeg_quality_estimate or 100) < classify.THRESHOLDS["jpegq_mild"]):
            try:
                from ..backends.direct import get_backend
                be = get_backend()
                if not be.is_loaded("fbcnn_color"):
                    tl = time.perf_counter()
                    be.get("fbcnn_color")
                    load_ms["fbcnn_color"] = (time.perf_counter() - tl) * 1e3
                qf = be.predict_jpeg_qf(img.rgb)
                if qf is not None:
                    dg.features["fbcnn_qf_pred"] = round(qf, 1)
            except Exception as e:
                log.warning("FBCNN QF prediction failed: %s", e)
        a = classify.assess(dg, summary, source_format=img.source_format, jpeg_quality=img.jpeg_quality_estimate,
                            png_text=md.png_text)
        t4 = time.perf_counter()
        rec = SourceRecord(path=p, sha256=sha, image=img, md=md, summary=summary, diag=dg, assessment=a,
                           dhash=dhash(img.rgb), rel_dir=rel_dir,
                           timings_ms={"hash": (t1 - t0) * 1e3, "metadata": (t2 - t1) * 1e3,
                                       "decode": (t3 - t2) * 1e3,
                                       "diagnose": (t4 - t3) * 1e3 - sum(load_ms.values()),
                                       **{f"model_load_{k}": v for k, v in load_ms.items()}})
        if record:
            st = p.stat()
            rec.source_id = self.db.upsert_source({
                "sha256": sha, "path": str(p), "rel_path": rel_dir, "format": img.source_format,
                "width": img.width, "height": img.height, "bit_depth": img.bit_depth,
                "has_alpha": int(img.alpha is not None), "size_bytes": st.st_size, "mtime": st.st_mtime,
                "dhash": f"{rec.dhash:016x}", "metadata_json": md.to_json(), "summary_json": summary,
                "diagnostics_json": dg.to_json(), "assessment_json": a.to_json()})
        return rec

    # ------------------------------------------------------------------ auto
    def request(self, mode: str | None = None, scale: float | str = "auto", target_long_edge: int | None = None,
                quality_override: bool | None = None, look: str | None = None) -> RouteRequest:
        return RouteRequest(mode=mode or self.settings.default_mode, scale=scale, target_long_edge=target_long_edge,
                            look=look or getattr(self.settings, "default_look", "enhanced"),
                            upscale_low_res_auto=self.settings.upscale_low_res_auto,
                            quality_override=self.settings.quality_override_default if quality_override is None
                            else quality_override, device_kind=self.device.kind, device_name=self.device.name,
                            allow_models=self.models_ok)

    def auto_recipe(self, src: SourceRecord, req: RouteRequest | None = None) -> Recipe:
        req = req or self.request()
        return route(src.assessment, req, src.image.width, src.image.height, self.profile)

    # ------------------------------------------------------------------ process
    def process(self, src: SourceRecord, recipe: Recipe, cancel: Callable[[], bool] | None = None,
                progress: Callable[[str, float], None] | None = None) -> ProcessResult:
        if recipe.no_op or not any(o.enabled for o in recipe.ops):
            return ProcessResult(rgb=src.image.rgb, alpha=src.image.alpha, recipe=recipe,
                                 timings_ms={"pipeline_total": 0.0})
        return self.pipeline.run(src.image.rgb, src.image.alpha, recipe, src.diag.faces, cancel=cancel,
                                 progress=progress, identity_check=self.settings.identity_check,
                                 fidelity_guard=self.settings.fidelity_guard)

    # ------------------------------------------------------------------ export
    def collection_name(self, batch_name: str | None) -> str:
        return batch_name or time.strftime("Single/%Y-%m-%d")

    def find_master(self, src: SourceRecord, recipe: Recipe) -> Path | None:
        row = self.db.one("SELECT path FROM intermediates WHERE key=?", (f"{src.sha256}:{recipe.hash}",))
        if row and Path(row["path"]).is_file():
            return Path(row["path"])
        return None

    def export(self, src: SourceRecord, rgb: np.ndarray, alpha: np.ndarray | None, recipe: Recipe,
               profile_key: str, *, job_id: int | None = None, batch_id: int | None = None,
               batch_name: str | None = None, review_flags: list[str] | None = None, elapsed_ms: float | None = None,
               overrides: dict[str, Any] | None = None, label: str = "", edited: bool = False,
               generative: bool | None = None, parent_output_id: int | None = None) -> OutputRecord:
        prof = profiles_from_settings(self.settings)[profile_key]
        if overrides:
            prof = ExportProfile(**{**prof.__dict__, **overrides})
        if edited and prof.kind == "restored_master":
            prof = ExportProfile(**{**prof.__dict__, "kind": "edited_variant"})
        dest = output_path(Path(self.settings.output_root), self.collection_name(batch_name), src.rel_dir,
                           src.path.name, src.sha256, prof.kind, recipe.hash, prof.fmt)
        gen = recipe.is_generative if generative is None else generative
        final, audit, (w, h, bits), data = write_output(
            rgb, alpha, src.image.icc_profile, prof, src.md, dest, source_bits=src.image.bit_depth,
            source_sha=src.sha256, recipe_hash=recipe.hash, recipe_summary=recipe.summary(),
            scale=recipe.target_scale, generative=gen, edited=edited or prof.kind in ("edited_variant", "cutout"),
            png_text=src.md.png_text)
        out_sha = sha256_bytes(data)
        models = sorted({o.model_id for o in recipe.ops if o.enabled and o.model_id})
        oid = self.db.insert("outputs", {
            "job_id": job_id, "source_id": src.source_id, "batch_id": batch_id, "kind": prof.kind,
            "profile": profile_key, "path": str(final), "sha256": out_sha, "format": prof.fmt, "width": w,
            "height": h, "bit_depth": bits, "recipe_hash": recipe.hash, "recipe_summary": recipe.summary(),
            "models": ",".join(models), "review_flags_json": review_flags or [], "audit_json": audit,
            "elapsed_ms": elapsed_ms, "created": time.time(), "parent_output_id": parent_output_id,
            "label": label, "approval": "pending" if review_flags else "auto"})
        if prof.kind == "restored_master":
            self.db.execute("INSERT OR REPLACE INTO intermediates(key, output_id, path, created) VALUES(?,?,?,?)",
                            (f"{src.sha256}:{recipe.hash}", oid, str(final), time.time()))
        return OutputRecord(oid, prof.kind, profile_key, final, out_sha, w, h, bits, audit)

    def write_sidecar(self, src: SourceRecord, outcome: dict[str, Any], batch_name: str | None) -> Path:
        dest = output_path(Path(self.settings.output_root), self.collection_name(batch_name), src.rel_dir,
                           src.path.name, src.sha256, "restored_master", outcome["recipe"]["hash"], "TIFF")
        dest = dest.parent.parent / "Recipes" / (dest.stem.replace("__master__", "__recipe__") + ".json")
        data = json.dumps(outcome, indent=2, default=str).encode("utf-8")
        return imageio.write_bytes_no_clobber(data, dest)

    # ------------------------------------------------------------------ one-shot prepare
    def prepare(self, path: str | os.PathLike, *, profiles: tuple[str, ...] = ("archive",),
                recipe: Recipe | None = None, req: RouteRequest | None = None, rel_dir: str = "",
                batch_id: int | None = None, batch_name: str | None = None, job_id: int | None = None,
                cancel: Callable[[], bool] | None = None, progress: Callable[[str, float], None] | None = None,
                src: SourceRecord | None = None) -> JobOutcome:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        if src is None:
            src = self.inspect(path, rel_dir=rel_dir)
        timings.update({f"inspect_{k}": v for k, v in src.timings_ms.items() if not k.startswith("model_load_")})
        timings.update({k: v for k, v in src.timings_ms.items() if k.startswith("model_load_")})
        auto = self.auto_recipe(src, req)
        rec = recipe or auto
        if rec.outside_fast_profile and not rec.quality_override and self.device.kind == "cuda" \
                and self.profile.certified(self.device.name):
            timings["total_job_ms"] = (time.perf_counter() - t_start) * 1e3
            return JobOutcome(src, rec, auto, None, [], timings, rec.flags, "needs_override")
        t0 = time.perf_counter()
        master_path = self.find_master(src, rec)
        res: ProcessResult | None = None
        reused = False
        if master_path is not None and not getattr(rec, "_force_rerun", False):
            m = imageio.load_image(master_path)
            res = ProcessResult(rgb=m.rgb, alpha=m.alpha, recipe=rec, timings_ms={"reused_master": 0.0})
            reused = True
        else:
            res = self.process(src, rec, cancel=cancel, progress=progress)
        timings["process_ms"] = (time.perf_counter() - t0) * 1e3
        timings.update({f"op_{k}": v for k, v in res.timings_ms.items()})
        timings.update({f"model_load_{k}": v for k, v in res.model_load_ms.items()})
        flags = list(dict.fromkeys(res.review_flags))
        t1 = time.perf_counter()
        outs: list[OutputRecord] = []
        for pk in profiles:
            if pk == "archive" and reused:
                row = self.db.one("SELECT * FROM outputs WHERE path=? ORDER BY id DESC", (str(master_path),))
                if row:
                    outs.append(OutputRecord(row["id"], row["kind"], "archive", Path(row["path"]), row["sha256"],
                                             row["width"], row["height"], row["bit_depth"],
                                             json.loads(row["audit_json"] or "{}"), reused_master=True))
                continue
            outs.append(self.export(src, res.rgb, res.alpha, rec, pk, job_id=job_id, batch_id=batch_id,
                                    batch_name=batch_name, review_flags=flags))
        if reused:
            timings["reused_master_path"] = str(master_path)  # type: ignore[assignment]
        timings["export_ms"] = (time.perf_counter() - t1) * 1e3
        timings["total_job_ms"] = (time.perf_counter() - t_start) * 1e3
        timings["model_load_total_ms"] = float(sum(res.model_load_ms.values()) +
                                               sum(v for k, v in src.timings_ms.items() if k.startswith("model_load_")))
        timings["warm_job_ms"] = timings["total_job_ms"] - timings["model_load_total_ms"]
        if res.peak_vram_mb is not None:
            timings["peak_vram_mb"] = res.peak_vram_mb
        identity_flags = (res.identity or {}).get("flags", [])
        state = "review" if (identity_flags or res.guard_actions or rec.confidence < 0.55 or
                             (rec.outside_fast_profile and rec.quality_override) or rec.uses_gan or
                             rec.uses_external) else "done"
        outcome = JobOutcome(src, rec, auto, res, outs, timings, flags, state)
        self.write_sidecar(src, self.outcome_json(outcome), batch_name)
        return outcome

    def outcome_json(self, o: JobOutcome) -> dict[str, Any]:
        return {"app_version": __version__, "source": {"path": str(o.source.path), "sha256": o.source.sha256,
                                                       "format": o.source.image.source_format,
                                                       "width": o.source.image.width, "height": o.source.image.height,
                                                       "bit_depth": o.source.image.bit_depth,
                                                       "decode_notes": o.source.image.decode_notes},
                "metadata_summary": o.source.summary, "assessment": o.source.assessment.to_json(),
                "diagnostics": o.source.diag.to_json(), "auto_recipe": o.auto_recipe.to_json(),
                "recipe": o.recipe.to_json(), "identity": o.result.identity if o.result else None,
                "guard_actions": o.result.guard_actions if o.result else [],
                "device": self.device.__dict__,
                "outputs": [{"kind": x.kind, "profile": x.profile, "path": str(x.path), "sha256": x.sha256,
                             "size": [x.width, x.height], "bits": x.bits, "metadata_audit": x.audit}
                            for x in o.outputs],
                "timings_ms": o.timings_ms, "review_flags": o.review_flags, "state": o.state,
                "source_unchanged": sha256_file(o.source.path) == o.source.sha256}

    # ------------------------------------------------------------------ learning signal
    def record_correction(self, src: SourceRecord, auto: Recipe, final: Recipe, choice: str) -> None:
        self.db.insert("corrections", {"source_id": src.source_id, "source_sha256": src.sha256,
                                       "content_class": src.assessment.content_class,
                                       "auto_recipe_json": auto.to_json(), "final_recipe_json": final.to_json(),
                                       "choice": choice, "features_json": src.assessment.features,
                                       "created": time.time()})
