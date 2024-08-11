import cv2
DEBUG = True
USE_CUDA = False

from tools import get_face_loc
import cv2
# import argparse, os, random
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from torchvision import  transforms
# import pandas as pd
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
        self.face_cascade = cv2.CascadeClassifier('eyecontactcnn/data/haarcascade_frontalface_default.xml')


def get_focus_rate(frame,bbox): #读入一张图片，返回一个浮点数，代表: 视线对准摄像头人数/人数 
    
    model_weight = "eyecontactcnn/data/model_weights.pkl"
    model = model_static(model_weight,USE_CUDA=USE_CUDA)
    model_dict = model.state_dict()
    if USE_CUDA:
        snapshot = torch.load(model_weight)
    else:
        snapshot = torch.load(model_weight, map_location=torch.device('cpu'))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    if USE_CUDA:
        model.cuda()
    model.train(False)

    if DEBUG:
        # cv2.imshow("frame",frame)
        red = Color("red")
        colors = list(red.range_to(Color("green"),10))
        font = ImageFont.truetype("eyecontactcnn/data/arial.ttf", 40)

    height, width, channels = frame.shape
    # bbox = get_face_loc(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if DEBUG:
        print(bbox) 
        print("\n\nbbox len:",len(bbox),sep="")
    # frame_cnt += 1

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
        # forward pass
        if USE_CUDA:
            output = model(img.cuda())
        else:
            output = model(img)
        score = F.sigmoid(output).item()
        ans.append(score)
        
        coloridx = min(int(round(score*10)),9)
        draw = ImageDraw.Draw(frame)
        drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
        draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)

        if DEBUG:
            cnt += 1

        # if DEBUG:
        #     coloridx = min(int(round(score*10)),9)
        #     draw = ImageDraw.Draw(frame)
        #     drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
        #     draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
        #     cnt += 1
            # face = np.asarray(face) # convert PIL image back to opencv format for faster display
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # cv2.imshow("face",face)
            # print(score)
            # if cv2.waitKey(0)  == ord(' '):       # 每帧图像显示20ms，等待用户按下空格键退出
            #     continue
    if DEBUG:
        # 用matplotlib展示已有的frame图像
        plt.imshow(frame)
        plt.show()
        frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./out.jpg",frame)
        print("\n\n",cnt)


    # return ans
    return ans,frame
def get_sheltered_rate(img,face_loc,num_people): 
    # face_loc=get_face_loc(img)
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
    # return
    # print(img)
    # return
    '''
        读入图片,人数，返回（0/1/2/3）。0:可以拍摄，1:有遮挡，2:未注视摄像头，3:未注视摄像头且有遮挡
        bonus：返回不符合条件的人的位置
        依赖函数：get_focus_rate()，
                 get_sheltered_rate(),
                 get_face_loc() #可能需要
    '''
    loc=get_face_loc(img)
    save_img(img)
    focus=get_focus_rate(img,loc)[0]
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

# def baodi():
    

if '__main__'==__name__:
    path='./'
    file_name='test.png'
    img=cv2.imread(file_name)
    print(main(img,1))
