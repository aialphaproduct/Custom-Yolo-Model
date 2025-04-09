# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

import numpy as np
import cv2
import streamlit as st
import tempfile
import os
import gdown  # Import gdown


# Function to download YOLOv5 weights from Google Drive
def download_weights():
    file_id = "10ygsxRHye1DNgpErQZ6NghVIPhat6-UO"
    output_path = "model/yolov5.weights"

    if not os.path.exists(output_path):
        st.info("📥 Đang tải yolov5.weights từ Google Drive...")
        file_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(file_url, output_path, quiet=False)
        st.success("✅ Đã tải xong yolov5.weights!")
    else:
        st.info("✔️ File yolov5.weights đã có sẵn.")

# Function to get permission for webcam access
def request_webcam_permission():
    """Requests permission to use the webcam."""
    # In a real-world scenario, you would use a more robust method to request
    # webcam permissions, which is outside the scope of this basic example.
    # This is a placeholder for the permission request.
    permission_granted = st.checkbox("Allow access to webcam?")
    return permission_granted


def main():

    st.title("Object Detection with YOLOv5 CPU LOADING")

    # Sidebar options
    source = st.sidebar.selectbox("Choose input source:", ("Image", "Video", "Webcam"))

    # Placeholder for displaying the output image/video
    output_placeholder = st.empty()

    # Download button placeholder
    download_button_placeholder = st.empty()

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


    # Generate a color palette for the bounding boxes.  More robust, handles any number of classes.
    def generate_colors(class_labels):
        num_classes = len(class_labels)
        colors = []
        for i in range(num_classes):
          r = int((i * 57 + 137) % 256)  # Distribute colors more evenly
          g = int((i * 93 + 201) % 256)
          b = int((i * 47 + 83) % 256)
          colors.append((r, g, b))
        return colors

    class_colors = generate_colors(class_labels)
    class_colors = np.array(class_colors)



    def process_image(img_to_detect):
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]

        # convert to blob
        img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (320, 320), swapRB=True, crop=False)

        # Loading pretrained model
        yolo_model = cv2.dnn.readNetFromDarknet('model/yolov5.cfg', 'model/yolov5.weights')

        # Get all layers
        yolo_layers = yolo_model.getLayerNames()
        yolo_output_layer = [yolo_layers[i - 1] for i in yolo_model.getUnconnectedOutLayers().flatten()]

        yolo_model.setInput(img_blob)
        obj_detection_layers = yolo_model.forward(yolo_output_layer)

        class_ids_list = []
        boxes_list = []
        confidences_list = []

        for object_detection_layer in obj_detection_layers:
            for object_detection in object_detection_layer:
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]

                if prediction_confidence > 0.20:
                    predicted_class_label = class_labels[predicted_class_id]
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))

                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        if len(max_value_ids) > 0:
            for max_valueid in max_value_ids.flatten():
                box = boxes_list[max_valueid]
                start_x_pt = box[0]
                start_y_pt = box[1]
                box_width = box[2]
                box_height = box[3]

                predicted_class_id = class_ids_list[max_valueid]
                predicted_class_label = class_labels[predicted_class_id]
                prediction_confidence = confidences_list[max_valueid]

                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height

                box_color = class_colors[predicted_class_id]
                box_color = [int(c) for c in box_color]  # Ensure color is a list of integers

                predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                print("predicted object {}".format(predicted_class_label))

                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        return img_to_detect



    if source == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_to_detect = cv2.imdecode(file_bytes, 1)
            processed_image = process_image(img_to_detect)
            output_placeholder.image(processed_image, channels="BGR", caption="Processed Image")

             # Download processed image
            _, buffer = cv2.imencode(".jpg", processed_image)  # Encode as JPG
            download_button_placeholder.download_button(
                label="Download Processed Image",
                data=buffer.tobytes(),  # Convert to bytes
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )


    elif source == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            vid = cv2.VideoCapture(tfile.name)

            # Get video properties for output
            fps = vid.get(cv2.CAP_PROP_FPS)
            frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4.
            temp_out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            out = cv2.VideoWriter(temp_out_file.name, fourcc, fps, (frame_width, frame_height))


            while True:
                ret, frame = vid.read()
                if not ret:
                    break

                processed_frame = process_image(frame)
                out.write(processed_frame) # Write the processed frame to the output video
                output_placeholder.image(processed_frame, channels="BGR", caption="Processed Video")

            vid.release()
            out.release()

            # Provide download link for processed video.
            with open(temp_out_file.name, "rb") as file:
                download_button_placeholder.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

            # Clean up temporary files
            os.remove(tfile.name)
            os.remove(temp_out_file.name)



    elif source == "Webcam":
        if request_webcam_permission():
            cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

            if not cap.isOpened():
                st.error("Cannot open webcam. Please check your webcam connection and permissions.")
                return

            # Get webcam frame properties (for output video if needed)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:  # Handle case where FPS cannot be retrieved
                fps = 24  # Use a reasonable default (e.g. 24 FPS)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Streamlit doesn't support continuous video display within the main area efficiently.
            # We will show the processed frame as a snapshot in each iteration
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam.")
                    break

                processed_frame = process_image(frame)
                output_placeholder.image(processed_frame, channels="BGR", caption="Processed Webcam Feed")


            cap.release()
        else:
             st.write("Webcam access denied.")

if __name__ == "__main__":
    # Create the 'model' directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    download_weights()  # Call download_weights() to download or check for the weights file.
    main()