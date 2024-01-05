import cv2
from PIL import Image,ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import facenet_pytorch
print(facenet_pytorch.__file__)

# 打开视频文件
cap = cv2.VideoCapture('video.mp4')

# 检查是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
mtcnn = MTCNN(keep_all=True)
print(fps)
while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:  # 如果无法获取帧，则跳出循环
        break

    boxes, _ = mtcnn.detect(frame)  # 使用MTCNN检测人脸
    frame_draw = frame.copy()  # 创建帧的副本
    for box in boxes:
        # print(box)
        cv2.rectangle(frame_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # 使用cv2.rectangle绘制矩形框

    cv2.imshow('Frame', frame_draw)  # 使用cv2.imshow显示帧

    # 如果按下'q'键，退出循环
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # 按q键退出循环，等待时间为1秒/fps
        break

    # 释放cap对象，关闭所有窗口
cap.release()
cv2.destroyAllWindows()



