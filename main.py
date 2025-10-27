import sys
from pathlib import Path

sys.path.insert(0, "src")

from generator import generate_document, generate_line
from uploader import upload_subset_to_hub


def run_full_pipeline(
    corpus_path: str,
    repo_id: str,
    num_single_images: int,
    num_doc_images: int,
    single_line_dir: str,
    doc_dir: str,
):
    """데이터 생성부터 Hub 업로드까지 전체 과정을 실행합니다."""

    print("=" * 50)
    print(" OCR 데이터셋 생성 및 업로드 파이프라인 시작 ".center(50, "="))
    print("=" * 50)

    # STEP 1: 단일 문장 이미지 생성
    single_line_output_dir = generate_line(
        corpus_path=corpus_path,
        num_images=num_single_images,
        output_dir=single_line_dir,
    )

    if single_line_output_dir is None:
        print("단일 문장 이미지 생성에 실패하여 파이프라인을 중단합니다.")
        return

    # STEP 2: 문서 이미지 생성
    doc_output_dir = generate_document(
        corpus_path=corpus_path, num_images=num_doc_images, output_dir=doc_dir
    )

    if doc_output_dir is None:
        print("문서 이미지 생성에 실패하여 파이프라인을 중단합니다.")
        return

    print("-" * 50)
    print(" Hugging Face Hub 업로드 시작 ".center(50, "-"))
    print("-" * 50)

    # STEP 3: 각 데이터셋을 다른 config으로 동일한 저장소에 업로드
    # 단일 문장 데이터셋 업로드 (config_name="single_line")
    upload_subset_to_hub(
        dataset_dir=single_line_output_dir, repo_id=repo_id, config_name="single_line"
    )

    # 문서 데이터셋 업로드 (config_name="document")
    upload_subset_to_hub(
        dataset_dir=doc_output_dir, repo_id=repo_id, config_name="document"
    )

    print("\n" + "=" * 50)
    print(" 모든 작업 완료! ".center(50, "="))
    print(f"Hub에서 데이터셋 확인: https://huggingface.co/datasets/{repo_id}")
    print("=" * 50)


if __name__ == "__main__":
    # --- 공통 설정 ---
    CORPUS_FILE_PATH = "corpus.txt"
    NUM_SENTENCES_FOR_CORPUS = 5000

    # --- 데이터셋 설정 (하나의 저장소 ID 사용) ---
    HF_REPO_ID = "junyeong-nero/synthetic-ocr-bench"

    # --- 생성할 이미지 개수 설정 ---
    NUM_SINGLE_LINE_IMAGES = 1000
    NUM_DOCUMENT_IMAGES = 100

    # --- 출력 디렉토리 설정 ---
    SINGLE_LINE_OUTPUT_DIR = "images_single_line"
    DOC_OUTPUT_DIR = "images_document"

    # ======================================================================
    # 실행할 작업을 선택하세요
    # ======================================================================

    # --- STEP 0: 코퍼스 파일 생성 (최초 한 번 또는 텍스트 변경 시 실행 필요) ---
    # Path(CORPUS_FILE_PATH).unlink(missing_ok=True) # 기존 파일 삭제 후 시작
    # create_corpus_from_wiki(
    #     output_path=CORPUS_FILE_PATH, num_sentences=NUM_SENTENCES_FOR_CORPUS
    # )

    # --- STEP 1 & 2 & 3: 전체 파이프라인 실행 ---
    if Path(CORPUS_FILE_PATH).exists():
        run_full_pipeline(
            corpus_path=CORPUS_FILE_PATH,
            repo_id=HF_REPO_ID,
            num_single_images=NUM_SINGLE_LINE_IMAGES,
            num_doc_images=NUM_DOCUMENT_IMAGES,
            single_line_dir=SINGLE_LINE_OUTPUT_DIR,
            doc_dir=DOC_OUTPUT_DIR,
        )
    else:
        print(
            f"⚠️ 코퍼스 파일 '{CORPUS_FILE_PATH}'을(를) 찾을 수 없습니다. STEP 0을 먼저 실행하세요."
        )
