import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")

# Streamlit App Title
st.title("Object Detection - Classroom Items")

# Option to upload or capture
option = st.radio("Choose Input Method:", ["Upload Image", "Capture from Webcam"])

def run_inference(image_np):
    results = model(image_np, save=False, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_np

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        result_img = run_inference(image_np)
        st.image(result_img, caption="Detected Image", channels="RGB")

elif option == "Capture from Webcam":
    capture_btn = st.button("Capture Image")
    if capture_btn:
        cap = cv2.VideoCapture(0)
        st.info("Press 'c' to capture and close webcam window")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        cap.release()
        cv2.destroyAllWindows()

        result_img = run_inference(frame)
        st.image(result_img, caption="Detected Image from Webcam", channels="BGR")
