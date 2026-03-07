# 1M Dataset Refactoring Plan

## Objective

현재 synthetic OCR generation pipeline을 1M 샘플 규모까지 안정적으로 확장할 수 있도록 리팩터링한다.

이 계획의 목표는 단순히 "돌아간다"가 아니라 아래 조건을 만족하는 운영 가능한 경로를 만드는 것이다.

- generation 중 메모리 사용량이 전체 샘플 수에 비례해 증가하지 않을 것
- 장시간 실행 도중 중단되어도 안전하게 resume 가능할 것
- output artifact가 shard 단위로 관리되어 파일시스템/업로드 부담이 제한될 것
- upload 단계가 generation과 분리되거나 최소한 streaming/batched 방식으로 동작할 것
- 1M 실행 전에 10k -> 100k -> 1M scale rehearsal이 가능할 것

## Current Blocking Issues

현재 코드에서 1M 실행을 막는 핵심 문제는 다음과 같다.

1. Generation metadata가 메모리에 누적됨
   - `src/generator/generator.py`
   - `src/generation/mixed.py`
   - `src/generator/base.py`
2. Upload path가 전체 metadata를 다시 메모리화하고 mixed split 시 이미지까지 temp dir로 복제함
   - `src/generation/hub_upload.py`
   - `src/generation/hub_dataset.py`
3. Generation-side checkpoint/resume이 없음
   - `src/generator/base.py`
   - `src/generation/mixed.py`
4. 이미지가 flat directory에 쌓여 shard 관리가 불가능함
   - `src/generator/base.py`
   - `src/generator/generator.py`
5. renderer hot path와 formula cache가 장시간 실행에서 비효율적이거나 위험함
   - `src/generator/markdown_renderers.py`
   - `src/generator/markdown_render_utils.py`
6. docs와 실제 동작이 일부 어긋남
   - `docs/generation.md`
   - `src/generator/generator.py`

## Refactoring Principles

### 1. Streaming First

어떤 단계도 전체 dataset을 Python list로 들고 있지 않는다.

- metadata는 sample 생성 직후 append-only로 기록한다
- aggregate stats는 in-memory full scan 대신 incremental accumulator로 계산한다
- upload용 record도 shard 단위 또는 batch 단위로 처리한다

### 2. Resume by Default

장시간 job은 실패를 전제로 설계한다.

- sample index 또는 shard index 단위 checkpoint를 남긴다
- 재실행 시 이미 완료된 shard를 skip할 수 있어야 한다
- partial shard는 감지하고 안전하게 재생성하거나 복구 가능해야 한다

### 3. Separate Generation from Publication

1M generation과 HF upload를 하나의 CLI 호출에서 동기적으로 끝내려 하지 않는다.

- generation 완료 산출물과 upload 산출물을 분리한다
- upload는 별도 command 또는 post-processing phase로 분리한다
- generation 중 네트워크 이슈가 전체 pipeline 실패로 이어지지 않게 한다

### 4. Shard-Oriented Layout

artifact, metadata, stats를 shard 단위로 저장한다.

예시:

```text
output/
  ko/
    images_mixed/
      manifest.json
      shards/
        shard-000000/
          metadata.jsonl
          stats.json
          images/
            markdown_00000000.png
            ...
        shard-000001/
          metadata.jsonl
          stats.json
          images/
            ...
```

### 5. Verify at Smaller Scales Before 1M

리팩터링 완료 후 바로 1M를 돌리지 않는다.

- 10k: 기능/format 검증
- 100k: memory, resume, shard integrity 검증
- 1M dry-run subset: long-running stability 검증
- 1M full run: 최종 운영 실행

## Target End State

리팩터링 후 기대하는 최종 구조는 아래와 같다.

### Generation phase

- `main.py generate`는 local shard generation만 담당
- `--upload`는 optional로 바꾸거나 제거
- `--resume`, `--shard-size`, `--start-shard`, `--end-shard` 같은 옵션 지원
- metadata는 shard별 `metadata.jsonl` append 방식 저장
- shard 완료 시 shard manifest/checkpoint 기록

### Upload phase

- `main.py generate-upload` 또는 `main.py publish-dataset` 같은 별도 커맨드 추가
- shard metadata를 순차적으로 읽어 split assignment 수행
- temp dir full copy 없이 shard 또는 batch 단위 업로드
- 필요하면 parquet/webdataset/HF-friendly intermediate format 도입

### Monitoring phase

- run manifest에 시작 시각, seed, lang, template config, shard progress, failed shard 정보 기록
- shard별 stats 저장
- 전체 summary는 shard stats를 reduce해서 계산

## Workstreams

## Workstream 1 - Streaming Metadata and Incremental Stats

### Goal

샘플 수와 무관하게 generation memory footprint를 bounded 상태로 유지한다.

### Changes

- `src/generator/base.py`
  - `save_metadata(metadata: List[...])` 구조를 append writer 기반으로 교체
  - run 시작 시 metadata writer open, sample마다 line append, 종료 시 flush/close
- `src/generator/generator.py`
  - `generate()`가 full metadata list를 반환하지 않도록 변경
  - sample 생성 후 callback 또는 writer에 바로 전달
- `src/generation/mixed.py`
  - `all_metadata` 제거
  - sample 생성 직후 shard metadata writer에 append
- `src/generator/realism_stats.py`
  - full metadata list 입력 대신 incremental accumulator API 추가
  - 예: `update(meta)`, `finalize()`

### Deliverables

- in-memory metadata accumulation 제거
- shard 단위 stats writer 또는 incremental stats accumulator 도입
- 기존 metadata schema 유지

### Validation

- 10k generation 중 RSS memory가 샘플 수에 따라 선형 증가하지 않는지 확인
- 생성된 `metadata.jsonl`가 기존 schema와 호환되는지 테스트

## Workstream 2 - Checkpoint and Resume

### Goal

long-running generation job이 중단되어도 재실행 가능하게 만든다.

### Changes

- 새로운 run manifest 도입
  - 예: `run_manifest.json`
  - 기록 정보: repo_id, lang, seed, output_dir, shard_size, completed_shards, failed_shards, current_shard
- shard 완료 시 atomic marker 생성
  - 예: `shard-000123/_SUCCESS`
- partial shard 감지 규칙 정의
  - `_SUCCESS` 없음 + metadata/images count mismatch -> invalid shard로 간주
- CLI 옵션 추가
  - `--resume`
  - `--shard-size`
  - `--max-shards`
  - `--skip-completed-shards`

### Deliverables

- generation resume logic
- shard integrity check
- safe restart behavior

### Validation

- 중간 강제 종료 후 `--resume`로 이어서 생성되는지 검증
- 재실행 시 완료 shard를 덮어쓰지 않는지 검증

## Workstream 3 - Sharded Artifact Layout

### Goal

1M 파일이 단일 디렉터리에 몰리지 않도록 저장 구조를 개편한다.

### Changes

- 이미지 저장 경로를 shard-aware하게 변경
- metadata도 shard별로 분리
- top-level manifest에서 shard 목록/범위를 관리
- mixed mode output layout을 명확히 문서화

### Deliverables

- shard directory layout
- deterministic file naming policy
- top-level manifest and shard manifests

### Validation

- shard 단위로 artifact count 확인 가능
- 특정 shard만 재생성/삭제 가능
- 100k 리허설 시 filesystem 성능 저하가 flat layout 대비 완화되는지 확인

## Workstream 4 - Upload Decoupling and Batched Publication

### Goal

upload가 generation memory/disk bottleneck이 되지 않도록 pipeline을 분리한다.

### Changes

- `src/pipeline.py`
  - generation 성공 후 자동 upload 제거 또는 default off로 변경
- `src/generation/hub_upload.py`
  - 전체 metadata load/shuffle 구조 제거
  - shard iterator 기반 split assignment 도입
- `src/generation/hub_dataset.py`
  - `pandas.DataFrame` 전체 materialization 제거 방향 검토
  - 가능한 경우 shard 단위 dataset build 후 push
  - 필요 시 intermediate parquet export 경로 추가
- publish command 추가
  - 예: `main.py publish`
  - generation output root를 입력으로 받아 upload만 수행

### Design Notes

- split assignment는 deterministic해야 함
- global `random.Random(42).shuffle(records)` 대신 sample index 또는 stable hash 기반 split 결정 권장
- upload 단계에서 image full-copy temp dir 사용은 제거 대상

### Deliverables

- decoupled publish command
- batched/sharded upload path
- deterministic split assignment design

### Validation

- 10k shard set 업로드 시 peak memory 추적
- upload rerun 시 duplicate/overwrite 없이 idempotent 하게 동작하는지 검증

## Workstream 5 - Renderer and Hot Path Efficiency

### Goal

long-running generation throughput를 높이고 renderer-related instability를 줄인다.

### Changes

- `src/generator/markdown_renderers.py`
  - renderer instance 재사용 가능성 검토
  - `html2image` 사용 시 browser lifecycle 비용 최소화
- `src/generator/markdown_render_utils.py`
  - `_FORMULA_IMAGE_CACHE`에 size cap/LRU 도입
  - formula rendering fallback path 성능 측정
- `src/generator/generator.py`
  - AST 생성이 반드시 generation hot path에 있어야 하는지 재검토
  - 필요 시 `GT_json` 생성 시점을 deferred post-process로 이동

### Deliverables

- bounded formula cache
- renderer lifecycle 개선안
- optional deferred ground-truth expansion 설계

### Validation

- 10k benchmark에서 samples/sec 비교
- long-run에서 cache size bounded 여부 확인

## Workstream 6 - CLI and Documentation Alignment

### Goal

실행 방법과 실제 동작이 일치하도록 정리한다.

### Changes

- `main.py`
  - generation/publish/resume 관련 CLI 정비
- `docs/generation.md`
  - shard layout, resume flow, publish flow 문서화
  - 실제 A4 scaling 동작과 문서를 일치시킴
- `docs/cli.md`
  - `--mixed`의 현재 의미 재정의
  - upload 분리 후 예시 명령 업데이트
- wrapper scripts
  - `scripts/synthesize/generate.sh`가 shard-aware 옵션을 전달하도록 수정

### Deliverables

- CLI help 업데이트
- generation guide 업데이트
- 운영 runbook 초안

### Validation

- 문서 예제 명령이 실제 코드 경로와 일치하는지 점검

## Suggested Implementation Order

### Phase 0 - Safety Baseline

목표: 현재 동작을 고정하고 리팩터링 중 회귀를 막는다.

- metadata schema snapshot test 추가
- output layout snapshot test 추가
- mixed/non-mixed small-run integration test 추가

### Phase 1 - Streaming Core

목표: full metadata accumulation 제거.

- metadata append writer 도입
- incremental stats 도입
- generator loop에서 metadata list 제거

### Phase 2 - Resume and Shards

목표: long-running generation 안정화.

- shard layout 도입
- run manifest/checkpoint 도입
- resume CLI 추가

### Phase 3 - Upload Decoupling

목표: generation과 publication 분리.

- pipeline automatic upload off
- publish command 추가
- batched upload path 구현

### Phase 4 - Performance Cleanup

목표: renderer/cache/AST 비용 정리.

- formula cache bounded
- html2image lifecycle 최적화
- deferred `GT_json` 여부 결정

### Phase 5 - Documentation and Ops Readiness

목표: 운영 가능한 형태로 마무리.

- docs update
- shard runbook 작성
- 10k/100k/1M rehearsal checklist 작성

## Test and Verification Plan

### Unit Tests

- metadata append writer correctness
- shard manifest parser
- resume decision logic
- deterministic split assignment
- formula cache eviction policy

### Integration Tests

- 1k mixed generation with shard size 100
- interrupted run -> resume -> completed output consistency
- publish command on pre-generated shards
- rerun on completed output root without corruption

### Scale Rehearsal

- 10k: correctness, artifact integrity, docs smoke test
- 100k: memory profile, disk layout validation, resume drill
- 250k: long-run stability and upload rehearsal
- 1M: production run

### Success Metrics

- generation peak memory가 dataset size 증가에 따라 급격히 증가하지 않을 것
- interrupted run recovery time이 전체 재실행보다 훨씬 짧을 것
- upload 단계에서 full temp duplication이 없을 것
- shard 단위로 failure isolation 가능할 것

## Risks During Refactor

- metadata schema 변경으로 downstream evaluation/upload 호환성 깨질 수 있음
- shard layout 변경으로 기존 scripts가 경로를 가정한 부분이 깨질 수 있음
- upload decoupling 과정에서 README generation 또는 split naming이 달라질 수 있음
- resume logic이 잘못되면 duplicate sample 또는 missing shard가 생길 수 있음

## Non-Goals

이번 계획에서 반드시 같이 하지 않아도 되는 항목:

- evaluation pipeline 전면 개편
- model inference batching 최적화
- 새로운 synthetic template family 추가
- UI/dashboard 구축

## Recommended First PR Breakdown

### PR 1

Streaming metadata writer + incremental stats

대상 파일:

- `src/generator/base.py`
- `src/generator/generator.py`
- `src/generation/mixed.py`
- `src/generator/realism_stats.py`
- 관련 테스트

### PR 2

Shard layout + manifest + resume CLI

대상 파일:

- `src/pipeline.py`
- `main.py`
- `src/generator/base.py`
- `src/generation/mixed.py`
- 신규 manifest helper module
- 관련 테스트

### PR 3

Decoupled publish command + batched upload

대상 파일:

- `src/pipeline.py`
- `src/generation/hub_upload.py`
- `src/generation/hub_dataset.py`
- `main.py`
- 관련 테스트

### PR 4

Renderer/cache optimization + docs alignment

대상 파일:

- `src/generator/markdown_renderers.py`
- `src/generator/markdown_render_utils.py`
- `docs/generation.md`
- `docs/cli.md`
- `scripts/synthesize/generate.sh`

## Exit Criteria

아래 조건을 모두 만족하면 1M production run 준비가 끝난 것으로 본다.

- 100k generation run이 resume 포함 시나리오에서 안정적으로 완료됨
- generation peak memory가 bounded 상태임
- shard integrity check가 통과함
- publish 단계가 generation과 독립적으로 재실행 가능함
- docs/runbook 기준으로 운영자가 수동 개입 없이 배치를 실행 가능함
- 1M rehearsal에서 critical issue가 재발하지 않음
