# import cv2
# from ultralytics import YOLO

# model = YOLO(r"D:\egg\runs\detect\train\weights\best.pt")
# results = model.predict(source="egg.jpg", conf=0.25)

# image = cv2.imread("egg.jpg")
# # # 遍歷偵測結果，畫出框
# for result in results:
#     boxes = result.boxes  # 偵測的邊界框
#     print(f"Number of detections: {len(boxes)}")  # 確認是否有偵測到物件
#     for box in boxes:
#         # 取得座標
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         conf = box.conf[0]  # 信心分數
#         cls = int(box.cls[0])  # 類別索引
#         label = f"{model.names[cls]} {conf:.2f}"  # 顯示類別名稱和信心分數

#         # 畫框與標籤
#         color = (0, 255, 0)  # 綠色框
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 繪製矩形框
#         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 繪製標籤

# # # 讀取並顯示圖片
# image = cv2.imread("egg.jpg")  # 替換為結果的存檔路徑
# print(f"Number of detections: {len(result.boxes)}")

# cv2.imshow("YOLO Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO("best.pt")



# 確認是否載入圖片
image = cv2.imread("egg2.jpg")
image = cv2.resize(image, (640, 640))
# 推論圖片
results = model.predict(source=image, conf=0.25)
# 遍歷每個偵測結果
for result in results:
    boxes = result.boxes  # 偵測的邊界框
    print(f"Number of detections: {len(boxes)}")  # 確認是否有偵測到物件
    for box in boxes:
        # 取得座標
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]  # 信心分數
        cls = int(box.cls[0])  # 類別索引
        label = f"{model.names[cls]} {conf:.2f}"  # 顯示類別名稱和信心分數

        # 畫框與標籤
        color = (0, 255, 0)  # 綠色框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 繪製矩形框
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 繪製標籤

# 顯示圖片
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
