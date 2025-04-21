import streamlit as st
import cv2      
import numpy as np 
from torchvision.transforms import Compose, Resize, ToTensor, Normalize   
import matplotlib.pyplot as plt  
from PIL import Image
from ultralytics import YOLO

print("Available GPU:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Version:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

st.title("Multi-Model Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#load yolo for object detection
model_yolo1 = YOLO("C:/Users/PC 40/object_detection_yolov8/objects in the classroom/runs/detect/train3/weights/best.pt")
model_yolo2 = YOLO('yolov8m.pt')
model_yolo2.to(device)

#load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

#preprocessing image-resize and normalize the image
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

#position of object whether it is at left, right or center of the image
def get_position(center, image_width):
    if center < image_width * 0.4:          #left
        return "left"
    if center >image_width * 0.6:           #right
        return "right"
    else: 
        return "center"

# Estimate scale
img = cv2.imread("blackchair2meter.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

# Depth estimation
with torch.inference_mode():
    depth_pred = midas(input_batch)
    depth_pred = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()    

# Normalize Depth Map convert PyTorch tensor(GPU) to Numpy array (CPU)
depth_map_scale = depth_pred.cpu().numpy()
depth_map_scale = (depth_map_scale - depth_map_scale.min()) / (depth_map_scale.max() - depth_map_scale.min())

# Object Detection (YOLOv8 format)
results = model_yolo2(img_rgb)  # Note: Don't wrap in list for YOLOv8
scale = None
object_name = None

# Process detections
for result in results:
    for box in result.boxes:
        if box.conf > 0.5:  # Confidence threshold
            # Extract bounding box coordinates
            left, top, right, bottom = box.xyxy[0].tolist()
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            
            # Extract depth value of object
            object_depth_value = depth_map_scale[top:bottom, left:right]
            estimeted_inverse_depth = np.median(object_depth_value)
            
            # Calculate scale
            scale = 2 * estimeted_inverse_depth
            object_name = model_yolo2.names[int(box.cls)]
            print(f"Scale: {scale:.4f}")
            print(object_name)
            break  
    break  
