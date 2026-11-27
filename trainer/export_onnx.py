import torch
from model import PixelTransformer

model = PixelTransformer()
model.load_state_dict(torch.load("pixel_model_real.pth"))
model.eval()

dummy_text = torch.randn(1, 384)

torch.onnx.export(
    model,
    dummy_text, 
    "pixel_model.onnx",
    input_names=["text_embedding"],
    output_names=["pixel_colors"],
    dynamic_axes={"text_embedding": {0: "batch"}, "pixel_colors": {0: "batch"}},
    opset_version=14,
    dynamo=False
)