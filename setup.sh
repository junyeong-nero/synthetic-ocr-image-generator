# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# huggingface
uv tool install "huggingface_hub"


# clone repo
git clone https://github.com/junyeong-nero/synthetic-ocr-generator.git
cd synthetic-ocr-generator

# venv
uv venv --python 3.12 --seed
source .venv/bin/activate

# install vllm
uv add torch transformers accelerate "huggingface_hub[cli]"
uv pip install vllm --torch-backend=auto
uv pip install flash-attn --no-build-isolation
