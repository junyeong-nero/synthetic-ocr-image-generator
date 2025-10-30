import sys
from pathlib import Path
import yaml
import logging
import argparse  # argparse 모듈 추가

sys.path.insert(0, "src")
from pipeline import pipeline


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Run the data processing pipeline with a specified config file."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",  # 기본값 설정
        help="Path to the configuration YAML file (default: config.yaml)",
    )

    args = parser.parse_args()
    config_path = args.config
    logging.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)  # 파일이 없으면 에러 메시지 출력 후 종료
    except Exception as e:
        logging.error(f"Error loading or parsing the config file: {e}")
        sys.exit(1)  # 기타 에러 발생 시 종료

    pipeline(**config)
