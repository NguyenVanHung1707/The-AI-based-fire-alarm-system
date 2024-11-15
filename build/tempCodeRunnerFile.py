import cv2
import numpy as np
from yolodetect import YoloDetect

# Khởi tạo video capture và mô hình YOLO
video = cv2.VideoCapture('E:/hung/prj/Tainhanh.net_YouTube_Bedroom-Fire-Test_Media_ezJ6SorlpJo_001_480p.mp4')  # Đọc video từ đường dẫn
model = YoloDetect(detect_class="fire")

# Kích hoạt chế độ phát hiện ngay lập tức
detect = True

while True:
    ret, frame = video.read()  # Đọc khung hình từ video
    if not ret:
        print("Không thể đọc khung hình từ video.")
        break

    # Nếu kích hoạt chế độ phát hiện, chạy phát hiện trên toàn khung hình
    if detect:
        frame = model.detect(frame=frame)

    # Hiển thị khung hình lên màn hình
    cv2.imshow("Fire Detection", frame)

    # Dừng video capture và đóng các cửa sổ hiển thị khi video kết thúc
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

# Dừng video capture và đóng các cửa sổ hiển thị
video.release()
cv2.destroyAllWindows()
