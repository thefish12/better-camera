#gradio==3.50.2
import gradio as gr
import main
def f(img,num_people,lim_focus,lim_shelter):
    output_text=""
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

demo = gr.Interface(
    fn=f,
    inputs=[
    gr.Image(source="webcam", streaming=True),
    gr.Number(label="照片人数",value=1),
    gr.Slider(0, 1.0, value=1.0, label="注视强度"),
    gr.Slider(0, 1.0, value=0, label="遮挡强度"),
    ],title="正在检测看镜头和人脸遮挡",
    outputs=["text"],
    live=True,
)
demo.launch()