#gradio==3.50.2
#export GRADIO_SERVER_NAME="0.0.0.0"
# GRADIO_SERVER_PORT=8443
import gradio as gr
import time
import main
import cv2
output=0
def f(video,num_people,lim_focus,lim_shelter):
    output_text=""
    cap=cv2.VideoCapture(video)
    # print(type(img))
    # return 1
    # print(lim_focus,lim_shelter)
    try:
        while(1):
            _=int(time.time()*5)
            if(_&1):
                continue;
            ret,img=cap.read()
            cimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if not ret:
                break
            output=main.main(img,num_people,lim_focus,lim_shelter)
            if output[0]==0:
                return [cimg,"success"]
    except:
        failimg=cv2.imread("test-zyq.jpg")
        return [failimg,"failed"]
def snap(image,video,a=0,b=0):
    return [image,video]

demo = gr.Interface(
    f,
    [gr.Video(sources=["","webcam"]),gr.Number(value=1,label="照片人数"),gr.Slider(0, 1.0, value=0.8, label="注视强度"),gr.Slider(0, 1.0, value=0.2, label="遮挡强度")],
    [gr.Image(label="one shot"),"text"],
    title="正在检测看镜头和人脸遮挡",
    live=True,
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,

)

demo.launch(share=True)
# demo.queue().launch()
