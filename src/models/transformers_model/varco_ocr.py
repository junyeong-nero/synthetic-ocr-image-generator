import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

image = Image.open("file:///path/to/image.jpg")

# Image upscaling for OCR performance boost
w, h = image.size
target_size = 2304
if max(w, h) < target_size:
    scaling_factor = target_size / max(w, h)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    image = image.resize((new_w, new_h))

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "<ocr>"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.float16)

generate_ids = model.generate(**inputs, max_new_tokens=1024)
generate_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
]
output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)
print(output)
