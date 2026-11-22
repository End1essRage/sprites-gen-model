import torch
from model import PixelTransformer

model = PixelTransformer(vocab_size=17)
model.load_state_dict(torch.load("pixel_model_real.pth"))
model.eval()

# Для Conv1d длина должна быть хотя бы context_size (16)
dummy_pixels = torch.randint(0, 16, (1, 20)) # Берем 20 пикселей для теста
dummy_text = torch.randn(1, 384)

torch.onnx.export(
    model,
    (dummy_pixels, dummy_text),
    "pixel_model.onnx",
    input_names=["pixel_tokens", "text_embedding"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_tokens": {0: "batch", 1: "seq_len"},
        "text_embedding": {0: "batch"},
        "logits": {0: "batch", 1: "seq_len_out"}
    },
    opset_version=14,
    do_constant_folding=True,
    dynamo=False
)
print("Модель экспортирована в pixel_model.onnx")