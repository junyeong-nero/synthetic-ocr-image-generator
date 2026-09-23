# Calibration Report: `real_world_v2`

Produced with `main.py distribution measure/compare` on 2026-09-23. Synthetic sets use `--lang ko` with the Korean WikiText corpus (`corpus import-wikitext --kowikitext-split dev --kowikitext-split test`).

## Reference Sets (real images)

| Set | Pages | What it is | Median DPI (A4) | Grayscale | Coloured background |
|---|---:|---|---:|---:|---:|
| XFUND zh+ja validation | 100 | Scanned business forms | 300 | 100% | 0% |
| OmniDocBench demo | 18 | 9 document types, mostly born-digital, 2 fuzzy scans | 194 | 44% | 11% |
| Combined | 118 | Both of the above | 300 | 92% | 2% |

All were fetched from public GitHub repositories. Hugging Face was not reachable from the calibration environment, so the full OmniDocBench (1,355 pages) was not used. Re-run with `--hf-dataset opendatalab/OmniDocBench` where it is reachable.

`w1_norm` is the Wasserstein-1 distance divided by the reference p05–p95 spread: under 0.1 is close, 0.1 to 0.3 is noticeable, above 0.3 is different. For `is_*` rows the medians columns show shares.

## v1 → v2 Against the Combined Set

**v1** (`real_world_v1`, 60 samples, placeholder text):

| Metric | Reference median | Candidate median | W1 (norm) | Candidate is | Verdict |
|---|---:|---:|---:|---|---|
| background_luma | 254.000 | 254.500 | 0.637 | higher | different |
| est_dpi_a4 | 299.879 | 155.925 | 0.614 | lower | different |
| laplacian_var | 3990.948 | 848.771 | 0.557 | lower | different |
| aspect_ratio | 1.415 | 1.073 | 0.538 | lower | different |
| contrast | 154.500 | 57.500 | 0.387 | lower | different |
| ink_ratio | 0.057 | 0.025 | 0.303 | lower | different |
| noise_sigma | 1.827 | 0.948 | 0.299 | lower | noticeable |
| is_grayscale | 0.915 | 0.667 | 0.246 | lower | noticeable |
| colorfulness | 0.069 | 5.630 | 0.245 | higher | noticeable |
| abs_skew_deg | 0.200 | 0.100 | 0.208 | lower | noticeable |
| is_colored_background | 0.017 | 0.133 | 0.113 | higher | noticeable |
| is_binary | 0.025 | 0.117 | 0.088 | higher | close |

**v2** (`real_world_v2`, 120 samples, Korean corpus):

| Metric | Reference median | Candidate median | W1 (norm) | Candidate is | Verdict |
|---|---:|---:|---:|---|---|
| background_luma | 254.000 | 254.000 | 0.402 | lower | different |
| est_dpi_a4 | 299.879 | 204.716 | 0.390 | lower | different |
| is_grayscale | 0.915 | 0.667 | 0.248 | lower | noticeable |
| laplacian_var | 3990.948 | 2667.454 | 0.231 | lower | noticeable |
| colorfulness | 0.069 | 5.215 | 0.217 | higher | noticeable |
| abs_skew_deg | 0.200 | 0.100 | 0.208 | lower | noticeable |
| ink_ratio | 0.057 | 0.062 | 0.147 | higher | noticeable |
| noise_sigma | 1.827 | 1.942 | 0.094 | higher | close |
| is_colored_background | 0.017 | 0.108 | 0.091 | higher | close |
| aspect_ratio | 1.415 | 1.411 | 0.085 | lower | close |
| contrast | 154.500 | 149.500 | 0.050 | lower | close |
| is_binary | 0.025 | 0.050 | 0.024 | higher | close |

The number of metrics rated "different" dropped from 6 to 2:
- Contrast went from 57.5 to 149.5 (real: 154.5).
- Aspect ratio went from 1.07 to 1.41 (real: 1.415).
- Ink coverage went from 2.5% to 6.2% (real: 5.7%).

The remaining DPI gap is intentional. The combined reference is 85% 300-dpi scans, while `real_world_v2` is 45% born-digital at 150–300 dpi.

## Per Channel

**Synthetic `scanned` (42 samples) vs XFUND scans:**

| Metric | Reference median | Candidate median | W1 (norm) | Candidate is | Verdict |
|---|---:|---:|---:|---|---|
| colorfulness | 0.043 | 0.000 | 0.460 | lower | different |
| est_dpi_a4 | 299.879 | 307.557 | 0.342 | higher | different |
| background_luma | 254.000 | 254.000 | 0.241 | lower | noticeable |
| laplacian_var | 3990.948 | 2387.964 | 0.214 | lower | noticeable |
| contrast | 151.000 | 119.500 | 0.104 | lower | noticeable |
| noise_sigma | 1.805 | 1.964 | 0.095 | higher | close |
| aspect_ratio | 1.415 | 1.410 | 0.078 | lower | close |
| abs_skew_deg | 0.200 | 0.225 | 0.070 | higher | close |
| is_grayscale | 1.000 | 0.929 | 0.066 | lower | close |
| is_colored_background | 0.000 | 0.071 | 0.066 | higher | close |
| ink_ratio | 0.053 | 0.057 | 0.052 | higher | close |
| is_binary | 0.030 | 0.024 | 0.012 | lower | close |

**Synthetic `born_digital` (56 samples) vs OmniDocBench demo:**

| Metric | Reference median | Candidate median | W1 (norm) | Candidate is | Verdict |
|---|---:|---:|---:|---|---|
| contrast | 175.500 | 152.000 | 0.384 | lower | different |
| ink_ratio | 0.107 | 0.057 | 0.310 | lower | different |
| colorfulness | 15.165 | 6.574 | 0.253 | lower | noticeable |
| is_grayscale | 0.444 | 0.661 | 0.220 | higher | noticeable |
| abs_skew_deg | 0.100 | 0.100 | 0.166 | lower | noticeable |
| aspect_ratio | 1.397 | 1.414 | 0.154 | higher | noticeable |
| noise_sigma | 2.919 | 2.144 | 0.151 | lower | noticeable |
| est_dpi_a4 | 193.712 | 204.474 | 0.147 | higher | noticeable |
| laplacian_var | 4060.004 | 3858.977 | 0.109 | lower | noticeable |
| background_luma | 255.000 | 255.000 | 0.104 | lower | noticeable |
| is_binary | 0.000 | 0.089 | 0.086 | higher | close |
| is_colored_background | 0.111 | 0.054 | 0.042 | lower | close |

## Known Remaining Gaps

- **Born-digital ink coverage** is 5.7% vs 10.7%, and more synthetic pages are grayscale (66% vs 44%). The OmniDocBench demo pages are dense CJK books and newspapers with colour figures and photos. The generator draws image blocks as grey placeholder boxes, and there are no multi-column layouts yet.
- **Scanned `colorfulness`** is flagged "different" only because both values are near zero (0.04 vs 0.00).
- **Reference size:** 118 pages is small and forms-heavy. Treat the numbers as a first fit and re-run the loop on the data you target.
