DEBUG = True
USE_CUDA = False
GRADIO_SERVER_PORT=8443

import gradio as gr
import random
from tools import get_face_loc
import cv2
import torch
import torch.nn.functional as F
from torchvision import  transforms
import numpy as np
from eyecontactcnn.model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
import matplotlib.pyplot as plt
def save_img(img):
    pass

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

class FaceDetector:

    def __init__(self):
        self.model_weight = "eyecontactcnn/data/model_weights.pkl"
        self.model = model_static(self.model_weight,USE_CUDA=USE_CUDA)
        model_dict = self.model.state_dict()
        if USE_CUDA:
            snapshot = torch.load(self.model_weight)
        else:
            snapshot = torch.load(self.model_weight, map_location=torch.device('cpu'))
        model_dict.update(snapshot)
        self.model.load_state_dict(model_dict)
        if USE_CUDA:
            self.model.cuda()
        self.model.train(False)
    def get_focus_rate(self,frame,bbox): #读入一张图片，返回一个浮点数，代表: 视线对准摄像头人数/人数 
        if DEBUG:
            red = Color("red")
            colors = list(red.range_to(Color("green"),10))
            font = ImageFont.truetype("eyecontactcnn/data/arial.ttf", 40)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if DEBUG:
            print(bbox) 
            print("\n\nbbox len:",len(bbox),sep="")
        test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        frame = Image.fromarray(frame)
        ans = []
        if DEBUG:
            cnt = 0
        for b in bbox:
            face = frame.crop((b))
            img = test_transforms(face)
            img.unsqueeze_(0)
            if USE_CUDA:
                output = self.model(img.cuda())
            else:
                output = self.model(img)
            score = F.sigmoid(output).item()
            ans.append(score)
            
            coloridx = min(int(round(score*10)),9)
            draw = ImageDraw.Draw(frame)
            drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
            draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)

            if DEBUG:
                cnt += 1
        if DEBUG:
            # 用matplotlib展示已有的frame图像
            plt.imshow(frame)
            plt.show()
            frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./out.jpg",frame)
            print("\n\n",cnt)
        return ans,frame
def get_sheltered_rate(img,face_loc,num_people): 
    if len(face_loc)==0:
        return 0
    maxx,maxy=0,0
    cimg=img.copy()
    for _ in face_loc:
        x1,y1,x2,y2=_
        maxx=max(maxx,x2)
        maxy=max(maxy,y2)
        cv2.rectangle(cimg,(x1,y1),(x2,y2),(0,0,255),2)
    mp=[]
    for i in range(maxx):
        tmp=[]
        for j in range(maxy):
            tmp.append(0)
        mp.append(tmp)
    for _ in face_loc:
        x1,y1,x2,y2=_
        for i in range(x1,x2):
            for j in range(y1,y2):
                mp[i][j]+=1
                
    rate=0;all=0
    for i in range(maxx):
        for j in range(maxy):
            if(mp[i][j]):
                all+=1;
            if mp[i][j]>1:
                rate+=1;
    cv2.imwrite("tmp.png",cimg);
    return (rate/all)
def main(img,num_people,lim_focus=0.6,lim_shelter=0):
    '''
        读入图片,人数，返回（0/1/2/3）。0:可以拍摄，1:有遮挡，2:未注视摄像头，3:未注视摄像头且有遮挡
        bonus：返回不符合条件的人的位置
        依赖函数：get_focus_rate()，
                 get_sheltered_rate(),
                 get_face_loc() #可能需要
    '''
    loc=get_face_loc(img)
    save_img(img)
    
    focus=solve.get_focus_rate(img,loc)[0]
    focus_rate=sum(focus)/num_people
    sheltered_rate=get_sheltered_rate(img,loc,num_people)
    print("focus_rate:",focus_rate,"sheltered_rate:",sheltered_rate)
    if focus_rate>=lim_focus and sheltered_rate<=lim_shelter:
        return 0
    elif focus_rate>=lim_focus and sheltered_rate>lim_shelter:
        return 1
    elif focus_rate<lim_focus and sheltered_rate<=lim_shelter:
        return 2
    elif focus_rate<lim_focus and sheltered_rate>lim_shelter:
        return 3
    return 114

def f(video,num_people,lim_focus,lim_shelter):
    # output_text=""
    # x= random.randint(1,100)
    # print(x)
    # return x
    n=random.randint(1,100)
    # print(n)
    # return n
    img = cv2.imread(video)
    output=main(img,num_people,lim_focus,lim_shelter)
    if output==0:
        output_text="可以拍摄"

    elif output==1:
        output_text="有遮挡"
    elif output==2:
        output_text="未注视摄像头"
    else:
        output_text="未注视摄像头且有遮挡"
    if(output==0):
        path="output/"
        cv2.imwrite(path+str(n)+'.jpg',img)
        output_text+="保存于"+str(n)+".jpg"
    else:
        path="output/"
        cv2.imwrite(path+str(n)+'bad.jpg',img)
        output_text+="保存于"+str(n)+".jpg"        
    return output_text,img
    
def snap(image,video,a=0,b=0):
    return [image,video]
solve=FaceDetector()
demo = gr.Interface(
    f,
    [gr.Image(sources=["webcam"], streaming=True, type = "filepath"),gr.Number(value=1,label="people"), gr.Slider(0, 1.0, value=0.8, label="注视强度"),gr.Slider(0, 1.0, value=0.2, label="遮挡强度")],
    ["text",gr.Image()],
    title="正在检测看镜头和人脸遮挡",
    live=True,
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,

)

demo.launch()
# demo.queue().launch()