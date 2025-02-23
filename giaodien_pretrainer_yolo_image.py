import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
import gdown  # Thư viện tải file từ Google Drive

# --- Tải yolov3.weights từ Google Drive nếu chưa có ---
def download_weights():
    file_id = "10ygsxRHye1DNgpErQZ6NghVIPhat6-UO"  # ID của file trên Google Drive
    output_path = "model/yolov3.weights"

    if not os.path.exists(output_path):
        st.info("📥 Đang tải yolov3.weights từ Google Drive...")
        file_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(file_url, output_path, quiet=False)
        st.success("✅ Đã tải xong yolov3.weights!")
    else:
        st.info("✔️ File yolov3.weights đã có sẵn.")

# Gọi hàm tải file trước khi load mô hình
download_weights()

# --- Load YOLO Model ---
@st.cache_resource
def load_yolo():
    try:
        model = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")
        layer_names = model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers().flatten()]
        return model, output_layers
    except Exception as e:
        st.error(f"🚨 Lỗi khi tải mô hình YOLO: {e}")
        return None, None

yolo_model, yolo_output_layers = load_yolo()

# --- Class Labels ---
class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

# Màu cho bounding boxes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype="uint8")

# --- UI Streamlit ---
st.title("🖼️ YOLO Object Detection")
st.write("Tải ảnh lên để nhận diện đối tượng bằng YOLOv3.")

uploaded_file = st.file_uploader("📤 Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and yolo_model:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.read())
    image_path = tfile.name

    with open(image_path, "rb") as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img_height, img_width = img.shape[:2]
    img_blob = cv2.dnn.blobFromImage(img, 0.003922, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(img_blob)
    detection_layers = yolo_model.forward(yolo_output_layers)

    detected_objects = {}

    for layer in detection_layers:
        for detection in layer:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[:4] * np.array([img_width, img_height, img_width, img_height])
                (centerX, centerY, width, height) = box.astype("int")
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))
                endX = startX + width
                endY = startY + height

                if class_id not in detected_objects or detected_objects[class_id]["confidence"] < confidence:
                    detected_objects[class_id] = {"box": (startX, startY, endX, endY), "confidence": confidence}

    for class_id, obj in detected_objects.items():
        (startX, startY, endX, endY) = obj["box"]
        confidence = obj["confidence"]
        color = [int(c) for c in colors[class_id]]
        label = f"{class_labels[class_id]}: {confidence:.2f}"

        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="📍 Kết quả nhận diện", use_column_width=True)

    for _ in range(3):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            break
        except PermissionError:
            time.sleep(2)
