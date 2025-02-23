# -*- coding: utf-8 -*-
"""
@author: luu thanh my
"""

import numpy as np
import cv2

# Tải ảnh cần phát hiện, lấy chiều rộng và chiều cao
img_to_detect = cv2.imread('images/testing/scene2.jpg')
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]

# Chuyển đổi ảnh thành blob để đưa vào mô hình
img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
# 0.003922 = 1/255; kích thước blob (416, 416) có thể thay đổi tuỳ theo yêu cầu

# Danh sách các nhãn lớp 80 đối tượng
class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# Khai báo danh sách các màu (sử dụng dấu ',' trong chuỗi, sau đó chuyển đổi thành mảng số nguyên)
class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors, (16, 1))  # Nhân bản màu cho đủ số lớp

# Tải mô hình YOLO đã được huấn luyện trước
yolo_model = cv2.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')

# Lấy tên các lớp trong mô hình YOLO
yolo_layers = yolo_model.getLayerNames()
# Lấy các lớp đầu ra (lớp cuối cùng của mạng) từ mô hình
yolo_unconnected_out_layers = yolo_model.getUnconnectedOutLayers()  # Thêm dấu () khi gọi hàm
yolo_output_layer = [yolo_layers[i - 1] for i in yolo_unconnected_out_layers.flatten()]

# Đưa blob đã xử lý vào mô hình và chạy forward để nhận kết quả
yolo_model.setInput(img_blob)
obj_detection_layers = yolo_model.forward(yolo_output_layer)

# Vòng lặp qua từng đầu ra của lớp
for object_detection_layer in obj_detection_layers:
    # Vòng lặp qua từng phát hiện
    for object_detection in object_detection_layer:
        
        # Lấy điểm số của tất cả các đối tượng trong hộp giới hạn (bắt đầu từ index 5)
        all_scores = object_detection[5:]
        predicted_class_id = np.argmax(all_scores)
        prediction_confidence = all_scores[predicted_class_id]
    
        # Chỉ xử lý những dự đoán có độ tin cậy hơn 20%
        if prediction_confidence > 0.20:
            # Lấy nhãn đối tượng được dự đoán
            predicted_class_label = class_labels[predicted_class_id]
            # Tính toán tọa độ hộp giới hạn trên ảnh gốc
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
            start_x_pt = int(box_center_x_pt - (box_width / 2))
            start_y_pt = int(box_center_y_pt - (box_height / 2))
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height
            
            # Lấy màu cho hộp giới hạn từ mảng màu
            box_color = class_colors[predicted_class_id]
            # Chuyển đổi màu từ numpy array sang list
            box_color = [int(c) for c in box_color]
            
            # In thông tin dự đoán ra console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            print("Đối tượng được dự đoán:", predicted_class_label)
            
            # Vẽ hình chữ nhật và văn bản lên ảnh
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

# Sau khi vẽ xong tất cả các đối tượng, hiển thị ảnh
cv2.imshow("Detect Output", img_to_detect)
# Hàm waitKey() để chờ phím nhấn, nếu giá trị là 0 thì chờ vô hạn thời gian
cv2.waitKey(0)
# Đóng tất cả các cửa sổ khi nhấn phím
cv2.destroyAllWindows()
