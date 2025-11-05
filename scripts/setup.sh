# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone repo
git clone https://github.com/junyeong-nero/synthetic-ocr-generator.git
cd synthetic-ocr-generator

# venv
uv venv --python 3.12 --seed
source .venv/bin/activate

uv tool install "huggingface_hub"
uv add torch transformers accelerate datasets huggingface_hub hf_transfer addict easydict einops gpustat flashinfer-python
uv pip install flash-attn --no-build-isolation

# install vllm
uv pip install vllm --torch-backend=auto
