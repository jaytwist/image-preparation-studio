# Auto Accuracy Benchmark

Synthetic fixtures with injected (known) degradations. Objective only; user preferences are tracked separately from recorded corrections.

Content-class accuracy: **95%**

| issue | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| jpeg_artifacts | 4 | 1 | 0 | 0.8 | 1.0 |
| noise | 2 | 0 | 1 | 1.0 | 0.667 |
| color_cast | 2 | 3 | 0 | 0.4 | 1.0 |
| faded | 1 | 0 | 0 | 1.0 | 1.0 |
| oversharpened | 1 | 0 | 1 | 1.0 | 0.5 |
| oversmoothed | 1 | 3 | 0 | 0.25 | 1.0 |
| low_resolution | 7 | 10 | 0 | 0.412 | 1.0 |

## Expectation checks (×1)

| item | truth issues | predicted | Auto ×1 | checks |
|---|---|---|---|---|
| oldphone_astronaut | color_cast, jpeg_artifacts, low_resolution, noise | color_cast, jpeg_artifacts, low_resolution | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=35] → white_balance[classical:neutral_cast|strength=0.39,gains=[0.9377, 1.0, 1.0434]] | {} |
| oldphone_group | jpeg_artifacts, low_resolution | jpeg_artifacts, low_resolution | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=28] | {} |
| dslr_iso6400 | noise | low_resolution, noise | faithful ×1: denoise[model:scunet_real_psnr] | {} |
| dslr_16bit_p3 | noise | low_resolution, noise | faithful ×1: denoise[model:scunet_real_psnr] | {} |
| dslr_clean |  | color_cast, low_resolution | No enhancement needed (no-op) | {'no_op': True} |
| portrait_clean |  | low_resolution | No enhancement needed (no-op) | {'no_op': True} |
| portrait_x2 | low_resolution | low_resolution, oversmoothed | No enhancement needed (no-op) | {} |
| portrait_x4 | low_resolution | low_resolution, oversmoothed | No enhancement needed (no-op) | {} |
| group_x2 | low_resolution | low_resolution | No enhancement needed (no-op) | {} |
| group_x4 | low_resolution | low_resolution, oversmoothed | No enhancement needed (no-op) | {} |
| ui_clean |  | low_resolution | No enhancement needed (no-op) | {'no_op': True} |
| ui_jpeg | jpeg_artifacts | jpeg_artifacts, low_resolution | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=51] | {} |
| ui_x2 | low_resolution | jpeg_artifacts, low_resolution | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=19] | {} |
| ai_scene |  |  | No enhancement needed (no-op) | {'no_op': True} |
| ai_portrait |  | blur_soft | No enhancement needed (no-op) | {'no_op': True} |
| ai_portrait_jpeg | jpeg_artifacts | jpeg_artifacts | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=59] | {} |
| edited_oversharp | oversharpened | color_cast, low_resolution, oversharpened | No enhancement needed (no-op) | {'forbid_kinds': True} |
| edited_oversmooth | oversmoothed | low_resolution, oversmoothed | No enhancement needed (no-op) | {'forbid_kinds': True} |
| edited_hdr | oversharpened | color_cast, low_resolution | No enhancement needed (no-op) | {'forbid_kinds': True} |
| scan_faded | color_cast, faded | color_cast, faded, low_resolution | faithful ×1: white_balance[classical:neutral_cast|strength=0.46,gains=[0.9305, 1.0, 1.0792]] → exposure[classical:levels|ev=0.0,black=0.104,white=0.743,gamma=1.0] | {} |

## Routing vs neighbourhood (ablations/substitutions), ×1

Auto ranked first in **87%** of items; mean composite regret **0.0053**.

| item | Auto | best candidate | Auto rank | regret |
|---|---|---|---|---|
| oldphone_astronaut | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=35] → white_balance[classical:neutral_cast|strength=0.39,gains=[0.9377, 1.0, 1.0434]] | auto | 1/4 | 0.0 |
| oldphone_group | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=28] | auto | 1/3 | 0.0 |
| dslr_iso6400 | faithful ×1: denoise[model:scunet_real_psnr] | auto | 1/5 | 0.0 |
| dslr_16bit_p3 | faithful ×1: denoise[model:scunet_real_psnr] | auto[denoise→drunet_color] | 2/5 | 0.0777 |
| dslr_clean | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| portrait_clean | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| ui_clean | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| ui_jpeg | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=51] | auto | 1/3 | 0.0 |
| ai_scene | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| ai_portrait | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| ai_portrait_jpeg | faithful ×1: jpeg_cleanup[model:fbcnn_color|qf=59] | noop | 3/3 | 0.0021 |
| edited_oversharp | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| edited_oversmooth | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| edited_hdr | No enhancement needed (no-op) | noop | 1/2 | 0.0 |
| scan_faded | faithful ×1: white_balance[classical:neutral_cast|strength=0.46,gains=[0.9305, 1.0, 1.0792]] → exposure[classical:levels|ev=0.0,black=0.104,white=0.743,gamma=1.0] | auto | 1/4 | 0.0 |

User correction records: 0