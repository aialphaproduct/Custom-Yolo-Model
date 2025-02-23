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

    # Lưu các bounding boxes theo class_id
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

                # Nếu class_id chưa có trong dict, khởi tạo danh sách mới
                if class_id not in detected_objects:
                    detected_objects[class_id] = []

                # Lưu bounding box vào danh sách tương ứng với class_id
                detected_objects[class_id].append({
                    "box": (startX, startY, endX, endY),
                    "confidence": confidence,
                    "center": (centerX, centerY)
                })

    # Lọc bounding box có confidence cao nhất cho từng đối tượng riêng lẻ
    final_objects = []
    for class_id, objects in detected_objects.items():
        # Sắp xếp bounding boxes theo confidence giảm dần
        objects = sorted(objects, key=lambda x: x["confidence"], reverse=True)

        # Nhóm bounding boxes theo vị trí gần nhau
        grouped_objects = []
        for obj in objects:
            centerX, centerY = obj["center"]

            # Kiểm tra xem có đối tượng nào trong nhóm gần với vị trí này không
            found = False
            for group in grouped_objects:
                existing_centerX, existing_centerY = group[0]["center"]
                distance = np.sqrt((centerX - existing_centerX) ** 2 + (centerY - existing_centerY) ** 2)

                if distance < 50:  # Ngưỡng để xác định các đối tượng giống nhau
                    found = True
                    break

            # Nếu chưa có trong nhóm, thêm vào danh sách
            if not found:
                grouped_objects.append([obj])

        # Chọn bounding box có confidence cao nhất từ mỗi nhóm
        for group in grouped_objects:
            best_object = max(group, key=lambda x: x["confidence"])
            final_objects.append(best_object)

    # Vẽ bounding boxes lên ảnh
    for obj in final_objects:
        (startX, startY, endX, endY) = obj["box"]
        confidence = obj["confidence"]
        class_id = next(key for key, val in detected_objects.items() if obj in val)  # Lấy class_id
        color = [int(c) for c in colors[class_id]]
        label = f"{class_labels[class_id]}: {confidence:.2f}"

        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="📍 Kết quả nhận diện", use_column_width=True)

    # Xóa file tạm thời sau khi hiển thị
    for _ in range(3):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            break
        except PermissionError:
            time.sleep(2)
