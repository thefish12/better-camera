import cv2
import gradio as gr
# 定义一个函数，用于处理摄像头的实时视频流Q
def process_video(video_path):# 打开摄像头
    cap = cv2.VideoCapture(video_path)
    while True:
# 读取摄像头的每一帧
        ret,frame = cap.read()
        if not ret:
            break# 显示画面(这里不显示，因为Gradio会自动展示)         
        cv2.imshow('frame',frame)# 等待一段时间，减少CPU使用率
        cv2.waitKey(1)
# 释放摄像头资源
    cap.release()
    # 关闭所有0penCV窗口
    cv2.destroyAllWindows()
    # 返回一个空字符串，因为Gradio需要一个返回值
    return ""

iface=gr.Interface(fn=process_video,inputs=gr.Image(source="webcam",capture=True),outputs="empty",live=True,title="Live Webcam")
iface.launch()