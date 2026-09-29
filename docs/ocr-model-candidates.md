# OCR Model Candidates

Survey date: 2026-09-29. Goal: shortlist OCR and document-parsing models released or substantially updated after the Feb 2026 leaderboard that are worth adding to `configs/models/`. Models that already have a YAML are excluded (section 5 lists two that changed under an existing config). Nothing was run locally. Every number below is copied from a model card, paper, or Hugging Face (HF) leaderboard and is self-reported unless stated.

## 1. Scope and method

**Sources and queries**

- HF API (`/api/models`). `search=` terms: ocr, document parsing, doc parse, markdown, pdf, layout, table recognition, formula, parser, parse, reader, korean/japanese/cjk/hangul, and family names (dots, hunyuan, mineru, paddleocr, olmocr, docling, monkeyocr, infinity-parser, logics-parsing, qianfan, nanonets, lighton, surya, sarashina, Qwen3.5, gemma-4, EXAONE, kanana, HyperCLOVAX, VARCO, Solar).
- Sorted by `likes`, `downloads` and `createdAt`; the `filter=ocr` tag (about 540 models created since 2026-01-15); `createdAt` listings for about 30 orgs (baidu, PaddlePaddle, opendatalab, rednote-hilab, dots-studio, datalab-to, tencent, nvidia, infly, zenosai, jinaai, sbintuitions, Qwen, google, LGAI-EXAONE, kakaocorp, upstage, NCSOFT and others).
- Cutoff: created or materially updated on or after 2026-01-15. Quantised, GGUF and MLX re-uploads and single-user fine-tunes were skipped.
- Benchmarks: model cards; HF eval-results (`?expand[]=evalResults`, all flagged unverified by HF); dataset leaderboards for MDPBench (3,400 pages, 17 languages including ja/ko, photographed and digital), ParseBench (English enterprise PDFs), olmOCR-bench, Real5-OmniDocBench. MDPBench per-language values were read from the maintainers' README table and cross-checked against HF eval-results.
- Web: WebFetch on arXiv, GitHub and vendor pages. WebSearch quota ran out mid-survey, so section 6 is thin.

**Conventions**

- Kind: E2E = one VLM call per page. Pipeline = layout or detection stage plus per-region calls, or det+rec.
- T4 assumption: 16 GB, no bf16, no FlashAttention-2. "fp16 GB" is params x 2. All models below ship bf16 weights, so fp16 numerics are an untested risk for every one of them.
- Effort: **Config** = YAML only (generic `TransformersVLM`). **Config+** = YAML plus one `SPECIALIZED_MODEL_REGISTRY` line. **Small wrapper** = new file in `src/models/transformers/` for prompt or post-processing. **Pipeline** = wrapper that drives layout plus region calls, or an external package or server.

**How a backend gets added (from the code)**

- `create_model` selects a class by substring match of `model_id` in `SPECIALIZED_MODEL_REGISTRY` (`src/models/registry.py`). No match falls back to `TransformersVLM` (`src/models/local/transformers_vlm.py`).
- `TransformersVLM` loads anything that works with `AutoProcessor` + `AutoModelForImageTextToText` (trust_remote_code on, `dtype: float16` honoured) and sends one image plus one text prompt through `apply_chat_template`. It cannot pass chat-template kwargs (for example `enable_thinking=False`), image-size kwargs, or convert JSON layout output to markdown.
- The `openai` backend accepts `api_base`, so a single-pass model served with `vllm serve` needs only a YAML. Multi-call pipelines and custom logits processors do not fit. vLLM on a T4 was not verified.
- `DotsOCR` and `PaddleOCR` wrappers hardcode `torch.bfloat16` on CUDA; that needs changing for a T4.
- A new `dependency_group` needs entries in both `[dependency-groups]` and `[tool.uv.conflicts]` (AGENTS.md). Qwen3.5-architecture models (`qwen3_5`) could reuse the existing `chandra-ocr` group (Chandra OCR 2 is also `qwen3_5`), with the transformers floor raised; cards quote 5.2.0 to 5.6.0.

## 2. Ranked summary

| Tier | HF id | Kind | Size (fp16 GB) | Licence | CJK evidence | Key benchmark | Effort |
|---|---|---|---|---|---|---|---|
| 1 | `rednote-hilab/dots.mocr` | E2E (layout JSON) | 3.0B (6.1) | MIT + acceptable-use agreement | MDPBench ja 75.0, ko 78.7 | MDPBench 80.5; olmOCR-bench 83.9 | Config+ |
| 1 | `PaddlePaddle/PaddleOCR-VL-1.6` | Pipeline (layout + VLM) | 0.96B (1.9) | Apache-2.0 | MDPBench ja 70.3, ko 86.3 | MDPBench 78.9; OmniDocBench v1.6 96.33 | Pipeline |
| 1 | `zenosai/MonkeyOCRv2-B-Parsing` | Pipeline (vLLM server) | 0.88B (1.8) | Apache-2.0 | MDPBench ja 71.9, ko 87.6 | MDPBench 83.3 | Pipeline |
| 1 | `sbintuitions/sarashina2.2-ocr` | E2E | 3.9B (7.8) | MIT | ja/en only; VJRODa CER 22.6 | VJRODa CER 22.6 (dots.ocr 40.1) | Small wrapper |
| 1 | `ATH-MaaS/OvisOCR2` | E2E | 0.85B (1.7) | Apache-2.0 | none stated | OmniDocBench v1.6 96.58 | Config |
| 2 | `infly/Infinity-Parser2-Flash` | E2E (layout JSON) | 2.2B (4.4) | Apache-2.0 | "multilingual", no ja/ko numbers | ParseBench 73.25; olmOCR-bench 86.0 | Small wrapper |
| 2 | `opendatalab/MinerU2.5-Pro-2605-1.2B` | Pipeline (two-step) | 1.16B (2.3) | Apache-2.0 | MDPBench ja 59.1, ko 77.6 | OmniDocBench v1.6 95.72 | Pipeline |
| 2 | `datalab-to/surya-ocr-2` | Pipeline (one VLM) | 0.69B (1.4) | modified OpenRAIL-M | internal: ja 86.2%, ko 86.7% | olmOCR-bench 83.3 | Pipeline |
| 2 | `nvidia/NVIDIA-Nemotron-Parse-2.0` | E2E (enc-dec) | 0.9B (1.8) | OpenMDW-1.1 | "gains on CJK", no numbers | ParseBench 0.639 (vendor) | Small wrapper |
| 2 | `baidu/Qianfan-OCR` | E2E | 4.74B (9.5) | Apache-2.0 | "192 languages"; CCOCR-multilan 76.7 | OmniDocBench v1.5 93.12 | Config |
| 2 | `tencent/WeVisDoc-4B` (and `-2B`) | E2E | 4.4B (8.9) / 2.4B (4.9) | Apache-2.0 | en/zh only | OmniDocBench v1.6 95.38 | Config |
| 2 | `baidu/Unlimited-OCR` | E2E (multi-page) | 3.34B, ~0.5B active (6.7) | MIT | none; DeepSeek-OCR lineage | OmniDocBench v1.6 93.92 | Small wrapper |
| 3 | `XingChen-AGI/TeleOCR`, `jinaai/jina-ocr-v1`, `KDLAI/KDL-Frontier-Parser-nano`, `Logics-MLLM/Logics-Parsing-V3`, `PaddlePaddle/HPD-Parsing`, `PaddlePaddle/PP-OCRv6_*`, `Qwen/Qwen3.5-4B` | see section 3 | | | | | |

Tier 1 = 5, tier 2 = 7, tier 3 = 7. Order within a tier is a judgement that weighs CJK evidence against integration effort, not a score. MonkeyOCRv2 has the best ko number but the highest effort, so it sits third.

## 3. Per-model notes

### Tier 1: add first

**rednote-hilab/dots.mocr** (mirrored as `dots-studio/dots.mocr`)
- Created 2026-03-19; 3.04B; same `DotsOCRForCausalLM` architecture as dots.ocr; 358k downloads, 181 likes. Licence: MIT plus a supplementary agreement (MIT prevails on conflict; adds acceptable-use terms such as no unauthorised bulk digitisation of publications).
- Run: `trust_remote_code`. The card says the local directory name must not contain periods (the repo's dots.ocr config already loads from `./weights/DotsOCR`). vLLM >= 0.11 recommended. A registry line `dots.mocr` -> `DotsOCR` is needed because `dots.mocr` does not contain the substring `dots.ocr`. Wrapper reuse unverified.
- MDPBench (maintainers' table): overall 80.5 (digital 90.5, photographed 77.2), ja 75.0, ko 78.7, zh 84.6. Highest ja among the open-weight entries in that table (Gemini-3-pro-preview: 74.8). Other: olmOCR-bench 83.9; OmniDocBench v1.5 text edit 0.031; ParseBench 55.8 (English, HF leaderboard).
- Output: layout JSON with bbox, category, text (HTML tables, LaTeX formulas). T4: 6.1 GB, eager attention.
- Sources: https://huggingface.co/rednote-hilab/dots.mocr, arXiv 2603.13032 (per card).

**PaddlePaddle/PaddleOCR-VL-1.6**
- Created 2026-05-27; 0.96B; Apache-2.0; 41k downloads, 505 likes. Successor to the configured PaddleOCR-VL-1.5.
- Run: official page-level path is `paddleocr[doc-parser]>=3.6.0` + `paddlepaddle-gpu==3.2.1` (`PaddleOCRVL(pipeline_version="v1.6")`: layout detection, then per-element VLM), or the same pipeline against a vLLM server. The card states its `transformers>=5.0.0` example supports only element-level recognition and spotting (prompts such as `OCR:`, `Table Recognition:`).
- Languages: 109 (card, includes ja/ko). MDPBench per HF eval-results (2026-08-15): 78.9 (digital 87.1, photographed 76.3), ja 70.3, ko 86.3, zh 85.9; the MDPBench README still shows an older 75.0. Also OmniDocBench v1.6 96.33; Real5-OmniDocBench 93.19 (rank 1 of 17 on the HF leaderboard; scan, warp, photo, skew); ParseBench 67.43.
- Effort: Pipeline; needs a new dependency group (existing `paddle-ocr-vl` group is transformers-only). T4: 1.9 GB.
- See section 5 item 3: the existing 1.5 config probably uses the wrong path.
- Sources: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6, arXiv 2606.03264.

**zenosai/MonkeyOCRv2-B-Parsing** (also `MonkeyOCRv2-S-Parsing`)
- Created 2026-07-11; 0.88B on HF (card: 0.7B = 0.1B ViT + 0.6B LLM); Apache-2.0 per HF metadata; 113k downloads, 15 likes. S variant: 0.78B, ja 69.9, ko 88.7, overall 82.5.
- Run: custom code; parsing needs a vLLM server (0.11.2, or 0.25.1 with DFlash and CUDA >= 12.9) plus `parse.py` from https://github.com/Yuliang-Liu/MonkeyOCRv2. `parse.py` has a layout-recognition stage and an optional `--end2end` flag, so the default looks like layout then per-region calls (from the script, not the card). A CPU path was added 2026-08-22. vLLM on T4 unverified.
- MDPBench (maintainers' table): overall 83.3 (digital 88.1, photographed 81.7), ja 71.9, ko 87.6, zh 83.6. Highest overall among the open-weight entries in that table; ko above Gemini-3-pro-preview (85.5). Trained on 17 languages (MonkeyDoc v2). Self-reported; absent from the HF eval-results leaderboard; no ParseBench or olmOCR-bench numbers found.
- Effort: Pipeline. T4: 1.8 GB weights.
- Sources: https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing, arXiv 2607.11562.

**sbintuitions/sarashina2.2-ocr** (Japanese only)
- Created 2026-03-22; 3.9B (SigLIP2 encoder + Sarashina2.2-3B-Instruct); MIT; 1.7k downloads, 33 likes.
- Run: `trust_remote_code`, `AutoModelForCausalLM`; card pins `transformers==4.57.1`; the chat message contains an image and no text prompt; card uses greedy decoding, `repetition_penalty=1.2`, `max_new_tokens=6000`.
- Languages: ja and en only (no Korean); built for vertical Japanese text. VJRODa (Japanese, vertical text and reading order; CER lower is better): 22.6 vs dots.ocr 40.1, gpt-5-mini 72.4, Qwen3.5-4B 86.1, LightOnOCR-2-1B 158. olmOCR-bench overall 0.683, below LightOnOCR-2 0.773 and dots.ocr 0.722 (card).
- Output: markdown, HTML tables, LaTeX, figures as `<bbox>[(x1, y1), (x2, y2)]</bbox>`. Effort: Small wrapper (promptless message, generation kwargs, strip bbox tags). T4: 7.8 GB.
- Source: https://huggingface.co/sbintuitions/sarashina2.2-ocr.

**ATH-MaaS/OvisOCR2**
- Created 2026-07-13; 0.85B, post-trained from Qwen3.5-0.8B (SFT, RL, OPD); Apache-2.0; 212k downloads, 475 likes. Org identity is not stated on the card.
- Run: card shows only `vllm==0.22.1`. The architecture is native `qwen3_5`, so `AutoModelForImageTextToText` should load it (unverified). The chat template defaults to non-thinking, so the generic wrapper suffices. The card prompt asks for `<img src="images/bbox_l_t_r_b.jpg" />` tags for figures, which the card code strips afterwards.
- Languages: not stated; no ja/ko evidence found; not listed on MDPBench. Benchmarks (card): OmniDocBench v1.6 96.58; PureDocBench Avg3 75.06; third-party: Dr.DocBench challenge 59.25, Wild_OmniDocBench 87.91 (both from the TeleOCR card).
- Effort: Config plus tag stripping. T4: 1.7 GB; Qwen3.5 Gated DeltaNet layers may fall back to slow kernels (unverified).
- Sources: https://huggingface.co/ATH-MaaS/OvisOCR2, arXiv 2607.13639.

### Tier 2: add next

**infly/Infinity-Parser2-Flash**
- Created 2026-02-27 (updated 2026-07-20); 2.2B, `qwen3_5`; Apache-2.0; 27k downloads, 38 likes. The Pro variant is a 35B MoE (rejected).
- Run: native transformers. The card example loads with `torch_dtype="float16"`, uses `qwen_vl_utils` and `enable_thinking=False`, and prompts for a JSON list of layout elements, so it needs JSON-to-markdown post-processing.
- Languages "en, zh, multilingual", no ja/ko numbers. ParseBench 73.25 (HF; card 72.2), olmOCR-bench 86.0, OmniDocBench v1.6 91.98 (card).
- Effort: Small wrapper. T4: 4.4 GB. https://huggingface.co/infly/Infinity-Parser2-Flash

**opendatalab/MinerU2.5-Pro-2605-1.2B**
- Created 2026-05-20; 1.16B, Qwen2-VL architecture; Apache-2.0; 118k downloads (2604 version: 193k), 84 likes.
- Run via `mineru-vl-utils` (`MinerUClient.two_step_extract`, then `json2md`); transformers >= 4.56; vLLM needs `MinerULogitsProcessor`.
- Card lists zh/en. MDPBench 71.0 (digital 86.2, photographed 66.1), ja 59.1, ko 77.6. OmniDocBench v1.6 95.72, ParseBench 72.78, Real5-OmniDocBench (2604 version) 88.94.
- Effort: Pipeline. T4: 2.3 GB. https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B

**datalab-to/surya-ocr-2**
- Created 2026-05-14; 0.69B, `qwen3_5`; 1.33M downloads, 118 likes. Weights: modified AI Pubs OpenRAIL-M (card: free for research, personal use and startups under $5M funding or revenue); code is Apache-2.0.
- Run: Surya 2 does layout, OCR and tables with one VLM served by vLLM or llama.cpp via `SuryaInferenceManager`. The repo's `SuryaOCREngine` uses the older `FoundationPredictor` API and would need rewriting. Output is HTML blocks with bbox.
- Internal 91-language pass rates: ja 86.2%, ko 86.7%, zh 82.5% (internal set, not comparable to the other numbers here). olmOCR-bench 83.3 (card), ParseBench 64.83.
- Effort: Pipeline. T4: 1.4 GB. https://huggingface.co/datalab-to/surya-ocr-2

**nvidia/NVIDIA-Nemotron-Parse-2.0**
- Repo created 2026-06-30 (card: released 2026-08-03); ~0.9B, C-RADIO ViT-H encoder + mBART decoder; OpenMDW-1.1 (card: commercial or non-commercial use), tokenizer CC-BY-4.0; 102k downloads, 117 likes. Different from the configured `nvidia/nemotron-ocr-v2`.
- Run: `transformers==5.6.1`, timm, open_clip_torch, einops, `trust_remote_code`, plus the repo's `postprocessing.py` (bbox, class and text to markdown, HTML, LaTeX). Input 1024x1280 to 1664x2048.
- Languages: "substantial gains on CJK and Indic-script" (card); evaluated on MOSCAR (includes Hangul, Japanese) but no per-language numbers. ParseBench 0.6391 vs 0.5782 for v1.2 (vendor).
- Effort: Small wrapper. T4: 1.8 GB in fp16 (safetensors stores F32). https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0

**baidu/Qianfan-OCR**
- Created 2026-03-18; 4.74B (Qwen3-4B decoder + Qianfan-ViT); Apache-2.0; 249k downloads, 1,204 likes.
- Run: native `qianfan_ocr` in transformers (docs page exists; minimum version unverified). Prompt "Parse this document to Markdown."; optional Layout-as-Thought (`enable_thinking=True`, up to 16k tokens).
- "192 languages", CCOCR-multilan 76.7, no ja/ko numbers. Card: OmniDocBench v1.5 93.12, olmOCR-bench 79.8, OCRBench 880. Third-party runs are lower: ParseBench 46.2, PureDocBench Clean 57.22 (TeleOCR card).
- Effort: Config. T4: 9.5 GB, tight with up to 4k visual tokens. https://huggingface.co/baidu/Qianfan-OCR, arXiv 2603.13398

**tencent/WeVisDoc-4B and -2B**
- Created 2026-09-16; Qwen3-VL-4B and -2B-Instruct fine-tunes (4.44B, 2.44B); Apache-2.0; 1.6k and 2.8k downloads, 49 and 26 likes. Output: markdown, HTML tables, LaTeX. The prompt lives in the GitHub client, not on the card (unverified).
- Card: en, zh only. OmniDocBench v1.6 95.38 / 95.06, PureDocBench Avg3 75.54 / 73.86. ParseBench 36.15 / 38.93 (layout score 0 because there is no bbox output, which depresses the mean).
- Effort: Config. A cheap test of whether a zh/en fine-tune keeps ko/ja, versus the configured Qwen3-VL-4B. T4: 8.9 / 4.9 GB. https://huggingface.co/tencent/WeVisDoc-4B

**baidu/Unlimited-OCR**
- Created 2026-06-19; 3.34B MoE (listed as "3B-A0.5B" in the WeVisDoc table, not on its own card). DeepSeek-OCR encoder/decoder with Reference Sliding Window Attention for multi-page parsing. MIT; 1.67M downloads, 4,312 likes.
- Run: custom `model.infer(...)`; card environment torch 2.10, transformers 4.57.1, CUDA 12.9. Prompts `<image>document parsing.` (single page) and `Multi page parsing.`; `no_repeat_ngram_size=35`. Output contains `<|det|>` tags (the card gives a `remove_det` helper).
- No language numbers; the DeepSeek-OCR base scores MDPBench ja 49.1 / ko 28.2. OmniDocBench v1.6 93.92 (WeVisDoc table), ParseBench 46.17.
- Effort: Small wrapper (adapt `deepseek_ocr.py`; needs its own dependency group because `deepseek-ocr` pins transformers 4.46.3). T4: 6.7 GB. https://huggingface.co/baidu/Unlimited-OCR, arXiv 2606.23050

### Tier 3: watch or skip

| HF id | Size, licence | Reason |
|---|---|---|
| `XingChen-AGI/TeleOCR` | 1.4B, Apache-2.0 | Two-stage (layout prompt, then per-crop prompts, OTSL tables) with an external GitHub pipeline; zh/en only; OmniDocBench v1.6 96.87, Real5 90.72, ParseBench 58.67. Revisit if ja/ko evidence appears. |
| `jinaai/jina-ocr-v1` | 3.4B, CC-BY-NC-4.0 | DeepSeek-OCR derivative; OmniDocBench v1.6 91.14, olmOCR-bench 83.4, ParseBench 45.93; non-commercial; "multilingual" unquantified. Research-only run possible. |
| `KDLAI/KDL-Frontier-Parser-nano` | 1.16B, AGPL-3.0 | Korean company, ko/en tags, ParseBench 76.36 (rank 1). Same architecture, parameter count (1,156,026,624) and prompts as MinerU2.5; the card does not name a base model (provenance unverified). AGPL. No Korean-specific numbers. |
| `Logics-MLLM/Logics-Parsing-V3` | 0.85B, none stated | Multi-page focus; created 2026-09-18; needs `vllm==0.22.1` + `transformers==5.6.0`; no CJK numbers. |
| `PaddlePaddle/HPD-Parsing` | 1.07B, Apache-2.0 | en/zh; needs a customised vLLM build; OmniDocBench v1.6 94.91. |
| `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}` | 1.5M to 34.5M, Apache-2.0 | Pipeline det+rec, 50 languages; text lines only, no tables or formulas, so it would repeat the low markdown scores of the configured `paddleocr` engines (ko 0.38 in Feb). Text-only baseline at most; needs PaddleOCR 3.x (unverified). |
| `Qwen/Qwen3.5-4B` (and 9B) | 4.66B, Apache-2.0 | General baseline. Thinking is on by default and the generic wrapper cannot set `enable_thinking=False`. Qwen3.5-9B on MDPBench: ja 55.7, ko 60.3, below Qwen3-VL-8B. 9B does not fit a T4 in fp16. |

## 4. Considered and rejected

- `infly/Infinity-Parser2-Pro`: 35.1B MoE (olmOCR-bench 87.6, ParseBench 74.3); does not fit a T4.
- `LGAI-EXAONE/EXAONE-4.5-33B`: Korean general VLM, 33B, no OCR numbers.
- `moonshotai/Kimi-K3` (2.78T), `Kimi-K2.5/2.6`, `zai-org/GLM-5.3-Flash` (321B), `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B` and `35B-A3B`, `deepseek-ai/DeepSeek-V4.1-Flash` (763B): general VLMs, too large. Reference only: Kimi-K3 MDPBench 83.6 (ja 74.9, ko 89.9).
- `google/gemma-4-{E2B,E4B,12B}-it`: general models, no OCR benchmark found.
- `tencent/Youtu-Parsing` (2.5B): licence "other", text says not intended for use in the EU; no ja/ko evidence.
- `numind/NuExtract3` (4.5B): structured extraction, not page parsing. Reconsider only if a KIE track is added.
- `ibm-granite/granite-vision-4.1-4b`: document understanding; ParseBench 39.45 (layout 0).
- `inclusionAI/ArmorOCR` (8.8B): adversarial and grounded OCR, en/zh.
- Community fine-tunes: `ebinan92/Qwen3.5-ocr-jp-2b` (ja/en, ruby and vertical text, no benchmarks), `tohoku-cijs/qwen3.5-2b-ndl-honkoku-kuzushiji-ocr` (historical script), `llm-jp/llm-jp-4-vl-9b` and `karakuri-ai/karakuri-vl-2-8b-thinking-2603` (general ja VLMs, no OCR numbers), `DocTron/OCRVerse` (no card).
- `kristaller486/dots.ocr-1.5`: third-party upload; `rednote-hilab/dots.ocr-1.5` returned no repo. `dots.mocr` is the current official release.
- `tencent/HunyuanOCR-1.5`: no separate repo; see section 5 item 1.

## 5. Open questions and unverified items

1. **`tencent/HunyuanOCR` changed under the existing config.** Since 2026-07-06 the repo root holds HunyuanOCR-1.5 (1.0 is under `v1.0/`, load with `subfolder="v1.0"`). Card: transformers >= 5.13.0, unified environment needs CUDA 13. The licence (Tencent Hunyuan Community License) excludes the EU, UK and South Korea, which matters for a Korean project. MDPBench: 1.5 = 76.8 (ja 65.5, ko 75.7), 1.0 = 68.3 (ja 55.6, ko 68.9). Pin a revision or re-run.
2. **Synthetic ranking vs MDPBench.** The Feb leaderboard ranks LightOnOCR-2-1B first on ja (0.978) and ko (0.974). MDPBench gives it ja 50.5, ko 41.9, overall 63.9 (digital 80.2, photographed 58.5), and DeepSeek-OCR ko 28.2, while dots.ocr scores ja 70.6, ko 68.5. The synthetic set may mostly measure the clean digital regime. `main.py rank-correlation` exists for this check but needs real-page scores (`docs/TODO.md`). Two external sets could help: MDPBench (public split) and `stockmark/OmniDocBench-JASyn` (518 synthetic Japanese pages, CC-BY-4.0, OmniDocBench format). `ONTHEIT/KDoc-OCRBench` (Korean, gated) was not readable.
3. **PaddleOCR-VL-1.5 config likely misuses the model.** `paddleocr-vl-1.5.yaml` sends a page-level markdown prompt through the element-level transformers path, which the card says supports only element recognition and spotting. The Feb score (ja 0.54, ko 0.54) may be a method artefact. Inference from the card, not tested.
4. **Benchmark comparability.** All numbers are self-reported; HF marks eval-results unverified. The MDPBench README and HF leaderboard disagree for PaddleOCR-VL-1.6 (75.0 vs 78.9). ParseBench is English only; OmniDocBench is mostly zh/en. MDPBench pages per language were not checked.
5. **Not verified:** vLLM and fp16 behaviour on a T4 for every model; whether Qwen3.5 Gated DeltaNet models run acceptably without extra kernels; minimum transformers versions for `qianfan_ocr` and `qwen3_5`; MonkeyOCRv2's default pipeline structure (inferred from `parse.py`); KDL-Frontier provenance; WeVisDoc prompt; whether `DotsOCR` works unchanged on dots.mocr.
6. **Licence audit before publishing results:** dots.mocr (acceptable-use agreement), Surya 2 (OpenRAIL-M limits), jina-ocr-v1 (non-commercial), KDL nano (AGPL), Tencent models (territory limits).

## 6. Non-HF and API models (out of HF scope, reference only)

- **Mistral OCR 4**, released 2026-06-23 per the vendor page (https://mistral.ai/news/ocr-4/): $4 per 1,000 pages ($2 batch), 170 languages. Japanese is named; Korean is not confirmed in the text fetched. Vendor-reported olmOCR-bench 85.20 and OmniDocBench 93.07. Successor to OCR 3 (Dec 2025).
- **MDPBench values for general APIs** (README table): Gemini-3-pro-preview 86.4 (ja 74.8, ko 85.5); Claude-Sonnet-4.6 73.1 (ja 63.4, ko 64.3); ChatGPT-5.2 68.6 (ja 55.8, ko 65.4). These are not the models configured here (gpt-5-mini, gemini-3-*, claude-opus-4-6 and sonnet-4-5) but give a scale for them.
- **Upstage Document Parse:** no version newer than the configured `document-parse-260128` was found; searches only surfaced Solar LLM releases (Solar Open 2, Solar Pro 4). Unverified.
