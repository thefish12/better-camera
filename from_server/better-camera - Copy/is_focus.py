DEBUG = False
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
import gc

# if DEBUG:
#     from PIL import ImageDraw
#     from PIL import ImageFont
#     from colour import Color
#     import matplotlib.pyplot as plt
#     def drawrect(drawcontext, xy, outline=None, width=0):
#         (x1, y1), (x2, y2) = xy
#         points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
#         drawcontext.line(points, fill=outline, width=width)

from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
import matplotlib.pyplot as plt
def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)




def get_focus_rate(frame): #读入一张图片，返回一个浮点数，代表: 视线对准摄像头人数/人数 
    
    model_weight = "./eyecontactcnn/data/model_weights.pkl"
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

    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("./eyecontactcnn/data/arial.ttf", 40)

    
    bbox = get_face_loc(frame)
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
        cv2.imwrite(".\out.jpg",frame)
        print("\n\n",cnt)


    # return ans
    del model
    gc.collect()
    return ans,frame

if __name__=="__main__":
    img = cv2.imread('test-zyq.jpg')
    if DEBUG:
        get_face_loc(img)
    print(get_focus_rate(img))
