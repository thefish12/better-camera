# import dlib
import cv2
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
args = parser.parse_args()


video_path = args.video

# set up video source
if video_path is None:
    cap = cv2.VideoCapture(0)
    video_path = 'live.avi'
else:
    cap = cv2.VideoCapture(video_path)


if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit()


# video reading loop
while(cap.isOpened()):
    ret, frame = cap.read()
    print("readed")
    if ret:
        cv2.imshow('114',frame)
        if cv2.waitKey(20)  == ord(' '):       # 每帧图像显示20ms，等待用户按下空格键退出
            break
        print("showing")

cap.release()
print('DONE!')


# if __name__ == "__main__":
#     run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)
