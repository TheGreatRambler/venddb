from ultralytics import YOLO

# Load the YOLO26 model
# Using the extra large model from https://docs.ultralytics.com/datasets/detect/coco/
# TODO use segmentation model
model = YOLO("yolo26x.pt")

# Export the model to ONNX format (creates `yolo26n.onnx`)
model.export(format="onnx")
