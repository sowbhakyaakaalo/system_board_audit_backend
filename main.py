from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load models once
battery_model = YOLO(os.path.join("models", "battery_cable.pt"))
led_model = YOLO(os.path.join("models", "led_board_cable.pt"))

@app.get("/")
async def root():
    return {"message": "✅ System Board Audit Backend is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    annotated_image = image.copy()

    component_results = {
        "battery": {"status": "Not Detected", "color": (128, 128, 128)},
        "led": {"status": "Not Detected", "color": (128, 128, 128)},
    }

    def detect(model, image, label_map, key):
        resized = cv2.resize(image, (640, 640))
        results = model.predict(resized, conf=0.25, iou=0.4, imgsz=640, verbose=False)

        highest_conf = 0.0
        selected_class = None
        selected_box = None

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = label_map[cls_id].lower().strip()
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            print(f"[{key.upper()}] → {class_name} (conf: {conf:.2f})")

            if conf > highest_conf:
                highest_conf = conf
                selected_class = class_name
                selected_box = xyxy

        if selected_class and selected_box:
            if selected_class in ["connect", "connected"]:
                component_results[key]["status"] = "Connected"
                component_results[key]["color"] = (0, 255, 0)
            elif selected_class in ["disconnect", "disconnected"]:
                component_results[key]["status"] = "Disconnected"
                component_results[key]["color"] = (0, 0, 255)

            color = component_results[key]["color"]

            # 🟩 Convert bbox back to original image size
            h_ratio = image.shape[0] / 640
            w_ratio = image.shape[1] / 640
            x1 = int(selected_box[0] * w_ratio)
            y1 = int(selected_box[1] * h_ratio)
            x2 = int(selected_box[2] * w_ratio)
            y2 = int(selected_box[3] * h_ratio)

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            label = f"{selected_class.capitalize()} {highest_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    detect(battery_model, image, battery_model.names, "battery")
    detect(led_model, image, led_model.names, "led")

    _, buffer = cv2.imencode('.png', annotated_image)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "image": encoded_img,
        "battery_status": component_results["battery"]["status"],
        "led_status": component_results["led"]["status"]
    })
