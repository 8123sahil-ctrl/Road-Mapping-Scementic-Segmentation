import torch
import segmentation_models_pytorch as smp

WEIGHTS = "deeplabv3plus_road_best.pth"
OUT_ONNX = "deeplabv3plus_road.onnx"

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,  # we load our trained weights
    in_channels=3,
    classes=2,
)

state = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,
    dummy,
    OUT_ONNX,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True
)

print("Saved:", OUT_ONNX)
