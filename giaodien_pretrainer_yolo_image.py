import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
import gdown  # Thư viện tải file từ Google Drive

# --- Tải yolov3.weights từ Google Drive nếu chưa có ---
def download_weights():
    file_id = "10ygsxRHye1DNgpErQZ6NghVIPhat6-UO"
    output_path = "model/yolov3.weights"

    if not os.path.exists(output_path):
        st.info("📥 Đang tải yolov3.weights từ Google Drive...")
        file_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(file_url, output_path, quiet=False)
        st.success("✅ Đã tải xong yolov3.weights!")
    else:
        st.info("✔️ File yolov3.weights đã có sẵn.")

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
st.write("Tải ảnh, video lên hoặc sử dụng webcam để nhận diện đối tượng bằng YOLOv3.")

# --- Sidebar để chọn chế độ ---
option = st.sidebar.selectbox("Chọn chế độ:", ["Ảnh tĩnh", "Video", "Webcam (Real-time)"])

if option == "Ảnh tĩnh":
    uploaded_file = st.file_uploader("📤 Chọn ảnh...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and yolo_model:
        # ... (Phần code xử lý ảnh tĩnh giữ nguyên) ...
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())
        image_path = tfile.name

        with open(image_path, "rb") as f:
             img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Giữ nguyên màu gốc

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

                    if class_id not in detected_objects:
                        detected_objects[class_id] = []

                    detected_objects[class_id].append({
                        "box": (startX, startY, endX, endY),
                        "confidence": confidence,
                        "center": (centerX, centerY)
                    })

        final_objects = []
        for class_id, objects in detected_objects.items():
            objects = sorted(objects, key=lambda x: x["confidence"], reverse=True)

            grouped_objects = []
            for obj in objects:
                centerX, centerY = obj["center"]

                found = False
                for group in grouped_objects:
                    existing_centerX, existing_centerY = group[0]["center"]
                    distance = np.sqrt((centerX - existing_centerX) ** 2 + (centerY - existing_centerY) ** 2)

                    if distance < 50:
                        found = True
                        break

                if not found:
                    grouped_objects.append([obj])

            for group in grouped_objects:
                best_object = max(group, key=lambda x: x["confidence"])
                final_objects.append(best_object)

        for obj in final_objects:
            (startX, startY, endX, endY) = obj["box"]
            confidence = obj["confidence"]
            class_id = next(key for key, val in detected_objects.items() if obj in val)
            color = [int(c) for c in colors[class_id]]
            label = f"{class_labels[class_id]}: {confidence:.2f}"

            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        st.image(img, caption="📍 Kết quả nhận diện", use_column_width=True, channels="BGR")  # Hiển thị ảnh với màu chính xác

        # --- Lưu ảnh đã xử lý vào tệp tạm thời ---
        temp_save_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_save_path, img)  # Giữ nguyên màu khi lưu

        # --- Tạo nút tải xuống ảnh ---
        with open(temp_save_path, "rb") as file:
            st.download_button(
                label="📥 Tải ảnh kết quả",
                data=file,
                file_name="yolo_detection_result.jpg",
                mime="image/jpeg"
            )

        # Xóa file ảnh tạm thời
        for _ in range(3):
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                if os.path.exists(temp_save_path):
                    os.remove(temp_save_path)
                break
            except PermissionError:
                time.sleep(2)

elif option == "Video":
    uploaded_file = st.file_uploader("📤 Chọn video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None and yolo_model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        video_capture = cv2.VideoCapture(tfile.name)

        # --- Placeholder cho video output ---
        video_placeholder = st.empty()

        # --- Nút download (khởi tạo trước, để cập nhật sau) ---
        download_button_placeholder = st.empty()

         # --- Ghi video đầu ra ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Hoặc *'XVID'
        temp_output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        out = cv2.VideoWriter(temp_output_path, fourcc, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))


        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break  # Kết thúc khi hết video

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_height, img_width = frame.shape[:2]
            img_blob = cv2.dnn.blobFromImage(frame, 0.003922, (416, 416), swapRB=True, crop=False)

            yolo_model.setInput(img_blob)
            detection_layers = yolo_model.forward(yolo_output_layers)

             # --- Phần xử lý detection và vẽ bounding box (tương tự như phần ảnh tĩnh) ---
            detected_objects = {}

            for layer in detection_layers:
               for detection in layer:
                  scores = detection[5:]
                  class_id = np.argmax(scores)
                  confidence = scores[class_id]
                  if confidence > 0.5:
                     box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                     (centerX, centerY, width, height) = box.astype("int")

                     startX = int(centerX - (width / 2))
                     startY = int(centerY - (height / 2))
                     endX = int(startX + width)
                     endY = int(startY + height)

                     if class_id not in detected_objects:
                        detected_objects[class_id]=[]
                     detected_objects[class_id].append({"box": (startX,startY, endX, endY), "confidence": float(confidence), "center" : (centerX, centerY)})
            final_objects = []
            for class_id, objects in detected_objects.items():
               objects = sorted(objects, key=lambda x: x["confidence"], reverse = True)
               grouped_objects = []
               for obj in objects:
                  centerX, centerY = obj["center"]
                  found=False
                  for group in grouped_objects:
                     existing_centerX, existing_centerY = group[0]["center"]
                     distance = np.sqrt((centerX - existing_centerX)**2 + (centerY- existing_centerY)**2)
                     if distance < 50: #ngưỡng khoảng cách
                        found = True
                        break
                  if not found:
                     grouped_objects.append([obj])
               for group in grouped_objects:
                  best_object = max(group, key=lambda x:x["confidence"])
                  final_objects.append(best_object)

            for obj in final_objects:
               startX, startY, endX, endY = obj["box"]
               confidence = obj["confidence"]
               class_id = next(key for key, value in detected_objects.items() if obj in value)
               color = [int(c) for c in colors[class_id]]
               label = "%s: %.2f" % (class_labels[class_id], confidence)
               cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
               cv2.putText(frame, label, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Hiển thị frame lên placeholder ---
            video_placeholder.image(frame, channels="RGB")

            # --- Ghi frame vào video đầu ra ---
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Ghi frame đã xử lý

        video_capture.release()
        out.release()

        # --- Cập nhật nút download ---
        with open(temp_output_path, "rb") as file:
            download_button_placeholder.download_button(
                label="📥 Tải video kết quả",
                data=file,
                file_name="yolo_detection_video.mp4",
                mime="video/mp4"
            )
        # --- Dọn dẹp tệp tạm ---
        try:
            os.remove(tfile.name)
            os.remove(temp_output_path)
        except Exception as e:
            st.error(f"Lỗi khi xóa tệp tạm thời: {e}")

elif option == "Webcam (Real-time)":
    st.write("### 📹 Nhận diện đối tượng qua Webcam")
    run = st.checkbox("▶️ Bắt đầu / Dừng")
    FRAME_WINDOW = st.image([])  # Placeholder để hiển thị khung hình
    camera = cv2.VideoCapture(0)  # Sử dụng webcam mặc định (index 0)

    while run and yolo_model:
        ret, frame = camera.read()
        if not ret:
            st.error("Không thể truy cập webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Chuyển sang RGB để hiển thị đúng màu
        img_height, img_width = frame.shape[:2]
        img_blob = cv2.dnn.blobFromImage(frame, 0.003922, (416, 416), swapRB=True, crop=False)

        yolo_model.setInput(img_blob)
        detection_layers = yolo_model.forward(yolo_output_layers)

        # --- Xử lý detection và vẽ bounding box ---
        detected_objects = {}

        for layer in detection_layers:
           for detection in layer:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.5:
                 box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                 (centerX, centerY, width, height) = box.astype("int")

                 startX = int(centerX - (width / 2))
                 startY = int(centerY - (height / 2))
                 endX = int(startX + width)
                 endY = int(startY + height)

                 if class_id not in detected_objects:
                    detected_objects[class_id]=[]
                 detected_objects[class_id].append({"box": (startX,startY, endX, endY), "confidence": float(confidence), "center" : (centerX, centerY)})
        final_objects = []
        for class_id, objects in detected_objects.items():
           objects = sorted(objects, key=lambda x: x["confidence"], reverse = True)
           grouped_objects = []
           for obj in objects:
              centerX, centerY = obj["center"]
              found=False
              for group in grouped_objects:
                 existing_centerX, existing_centerY = group[0]["center"]
                 distance = np.sqrt((centerX - existing_centerX)**2 + (centerY- existing_centerY)**2)
                 if distance < 50: #ngưỡng khoảng cách
                    found = True
                    break
              if not found:
                 grouped_objects.append([obj])
           for group in grouped_objects:
              best_object = max(group, key=lambda x:x["confidence"])
              final_objects.append(best_object)

        for obj in final_objects:
           startX, startY, endX, endY = obj["box"]
           confidence = obj["confidence"]
           class_id = next(key for key, value in detected_objects.items() if obj in value)
           color = [int(c) for c in colors[class_id]]
           label = "%s: %.2f" % (class_labels[class_id], confidence)
           cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
           cv2.putText(frame, label, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        FRAME_WINDOW.image(frame)  # Cập nhật khung hình lên placeholder

    camera.release()
    st.write("Kết thúc.")