import cv2

###占用 0号camera
camera = cv2.VideoCapture(0)
while(camera.isOpened()):
####start code
         result, frame =   camera.read()                     # 读取摄像头采集的图像   
         if result:
             cv2.imshow("camera",frame)             # 显示该图像
    
             if cv2.waitKey(20)  == ord(' '):       # 每帧图像显示20ms，等待用户按下空格键退出
                break

#end code
camera.release()                           # 释放摄像头对象
cv2.destroyAllWindows()