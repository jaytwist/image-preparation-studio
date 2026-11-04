# Model comparison — tuning corpus (synthetic fixtures with ground truth, CPU)

Objective full-reference metrics vs ground truth. *face sim* = ArcFace cosine vs the ground-truth face (worst face).
Times are on a 4-vCPU container running other jobs concurrently — **not** indicative of RTX 5070 Ti speed.
Visual evidence: `reports/eval/visual/*.png` (local, not committed). Sorted by SSIM.

## upscale_x2 · portrait_x2

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_x2 | 30.94 | 0.9608 | 1.59 | 1.13 | 0.972 | 23.5 |
| lanczos | 30.82 | 0.9526 | 1.66 | 0.46 | 0.972 | 0.0 |
| realesrgan_x4plus | 28.65 | 0.9275 | 2.69 | 0.87 | 0.913 | 11.5 |
| realesr_general_x4v3 | 28.40 | 0.9243 | 2.55 | 0.94 | 0.930 | 1.2 |
| realesrgan_x2plus | 27.12 | 0.8986 | 2.87 | 0.94 | 0.808 | 3.8 |
| realplksr_nomos2_x4 | 28.01 | 0.8820 | 2.59 | 0.97 | 0.960 | 9.2 |

## upscale_x2 · group_x2

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_x2 | 37.51 | 0.9823 | 0.71 | 0.94 | 0.984 | 316.1 |
| lanczos | 36.98 | 0.9782 | 0.73 | 0.59 | 0.986 | 0.1 |
| realplksr_nomos2_x4 | 30.14 | 0.9518 | 2.46 | 1.37 | 0.940 | 49.3 |
| realesrgan_x4plus | 30.16 | 0.9486 | 2.50 | 1.25 | 0.914 | 125.6 |
| realesr_general_x4v3 | 29.41 | 0.9424 | 2.39 | 1.51 | 0.915 | 9.4 |
| realesrgan_x2plus | 29.71 | 0.9361 | 2.50 | 1.20 | 0.883 | 69.2 |

## upscale_x2 · ui_x2

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| realplksr_nomos2_x4 | 21.41 | 0.9299 | 4.88 | 0.87 |  | 21.0 |
| swinir_x2 | 23.56 | 0.9126 | 1.36 | 0.90 |  | 73.8 |
| realesr_general_x4v3 | 22.03 | 0.9092 | 2.27 | 1.00 |  | 5.1 |
| realesrgan_x4plus | 22.57 | 0.9031 | 2.38 | 0.81 |  | 50.9 |
| nearest | 22.41 | 0.8926 | 1.48 | 0.63 |  | 0.0 |
| realesrgan_x2plus | 22.00 | 0.8880 | 2.48 | 0.70 |  | 28.8 |
| lanczos | 22.94 | 0.8742 | 1.74 | 0.39 |  | 0.0 |

## upscale_x4 · portrait_x4

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_x2+lanczos | 26.76 | 0.8938 | 2.47 | 0.58 | 0.700 | 21.0 |
| lanczos | 25.68 | 0.8510 | 2.79 | 0.20 | 0.750 | 0.0 |
| realesr_general_x4v3 | 23.61 | 0.8254 | 3.96 | 0.72 | 0.545 | 3.1 |
| realesrgan_x4plus | 23.68 | 0.8153 | 4.09 | 0.87 | 0.473 | 25.7 |
| realplksr_nomos2_x4 | 22.65 | 0.7854 | 4.26 | 0.97 | 0.666 | 10.5 |

## upscale_x4 · group_x4

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_x2+lanczos | 31.48 | 0.9262 | 1.35 | 0.63 | 0.791 | 96.6 |
| lanczos | 29.70 | 0.8956 | 1.61 | 0.26 | 0.769 | 0.1 |
| realplksr_nomos2_x4 | 26.31 | 0.8831 | 3.29 | 1.31 | 0.697 | 6.7 |
| realesrgan_x4plus | 27.05 | 0.8790 | 3.34 | 0.99 | 0.670 | 23.8 |
| realesr_general_x4v3 | 26.64 | 0.8754 | 3.25 | 1.01 | 0.684 | 1.0 |

## restore_noise · dslr_iso6400

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| scunet_real_psnr | 32.51 | 0.9160 | 1.79 | 0.66 |  | 25.6 |
| swinir_dn15 | 29.02 | 0.7384 | 2.67 | 1.79 |  | 276.1 |
| nlmeans | 25.98 | 0.6948 | 3.92 | 1.66 |  | 0.3 |
| drunet(σ=est) | 27.23 | 0.6402 | 3.22 | 2.25 |  | 16.0 |
| none | 25.47 | 0.5529 | 4.18 | 2.78 |  | 0.0 |

## restore_jpeg · oldphone_group

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_car_q30 | 32.19 | 0.9644 | 2.53 | 0.75 | 0.704 | 163.6 |
| fbcnn(qf=28) | 31.78 | 0.9624 | 2.67 | 0.73 | 0.700 | 9.2 |
| fbcnn(blind) | 31.78 | 0.9624 | 2.67 | 0.73 | 0.700 | 9.0 |
| swinir_car_q10 | 29.61 | 0.9340 | 3.44 | 0.65 | 0.498 | 214.0 |
| none | 28.80 | 0.9308 | 4.02 | 0.93 | 0.667 | 0.0 |

## restore_jpeg · ui_jpeg

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| swinir_car_q30 | 37.51 | 0.9970 | 0.89 | 1.04 |  | 291.8 |
| fbcnn(qf=41) | 35.57 | 0.9933 | 1.00 | 1.01 |  | 13.3 |
| fbcnn(blind) | 35.57 | 0.9933 | 1.00 | 1.01 |  | 13.1 |
| swinir_car_q10 | 32.05 | 0.9900 | 1.26 | 1.07 |  | 292.6 |
| none | 31.24 | 0.9505 | 1.90 | 1.18 |  | 0.0 |

## restore_jpeg · ai_portrait_jpeg

| candidate | PSNR | SSIM | ΔE | texture vs GT | face sim | CPU s |
|---|---|---|---|---|---|---|
| fbcnn(qf=59) | 40.12 | 0.9873 | 1.25 | 0.78 | 0.995 | 85.1 |
| fbcnn(blind) | 40.13 | 0.9873 | 1.25 | 0.79 | 0.995 | 113.4 |
| none | 37.35 | 0.9797 | 1.66 | 1.10 | 0.994 | 0.0 |
| swinir_car_q10 | 34.99 | 0.9611 | 2.07 | 0.65 | 0.938 | 1170.6 |
