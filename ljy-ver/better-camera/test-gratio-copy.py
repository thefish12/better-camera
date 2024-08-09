#gradio==3.50.2
#export GRADIO_SERVER_NAME="0.0.0.0"
# GRADIO_SERVER_PORT=8443


# import gradio as gr
# import time
# import main
# output=0
# async def f(img,num_people,lim_focus,lim_shelter):
#     def myoutput(img,num_people,lim_focus,lim_shelter) -> str:
#         output_text_dict = {
#         0: "可以拍摄",
#         1: "有遮挡",
#         2: "未注视摄像头",
#         3: "未注视摄像头且有遮挡"
#         }
#         return output_text_dict.get(main.main(img,num_people,lim_focus,lim_shelter), "未知结果")

#     # while True:
#     global output
#     x=int(time.time())
#     if(x%2==0):
#         return myoutput(img,num_people,lim_focus,lim_shelter)


# demo = gr.Interface(
#     f,
#     [
#     gr.Image(sources=["webcam"],streaming=True),
#     gr.Number(label="照片人数",value=1),
#     gr.Slider(0, 1.0, value=0.8, label="注视强度"),
#     gr.Slider(0, 1.0, value=0.2, label="遮挡强度"),
#     ],
#     ["text"],
#     live=True,
# )

# # demo.launch(share=True)
# demo.launch()






# import gradio as gr
# import numpy as np
# import time

# def flip(im):
#     im = np.flipud(im)
#     print(time.time())
#     return time.time()

# demo = gr.Interface(
#     flip, 
#     gr.Image(sources=["webcam"], streaming=True), 
#     "text",
#     live=True
# )
# demo.launch()
    




# import gradio as gr
# # from transformers import pipeline
# import numpy as np
# import time

# # transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# def transcribe(stream, new_chunk):
#     print(new_chunk)
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     print(time.time())
#     return stream, time.time()


# demo = gr.Interface(
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,
# )

# demo.launch()

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
    
    # print(type(img))
    # return 1
    while(1):
        img = cv2.imread(video)
        output=main.main(img,num_people,lim_focus,lim_shelter)
        if output==0:
            output_text="可以拍摄"
        elif output==1:
            output_text="有遮挡"
        elif output==2:
            output_text="未注视摄像头"
        else:
            output_text="未注视摄像头且有遮挡"
        
    return output_text
def snap(image,video,a=0,b=0):
    return [image,video]

demo = gr.Interface(
    f,
    [gr.Image(sources=["webcam"], streaming=True, type = "filepath"),gr.Number(value=1,label="people"), gr.Slider(0, 1.0, value=0.8, label="注视强度"),gr.Slider(0, 1.0, value=0.2, label="遮挡强度")],
    ["text"],
    title="正在检测看镜头和人脸遮挡",
    live=True,
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,

)

demo.launch()
# demo.queue().launch()