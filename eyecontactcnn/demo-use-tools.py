# import dlib
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
from tools import get_face_loc
import time 

import intel_extension_for_pytorch as ipex


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')
parser.add_argument('-cuda', help='use cuda', action='store_true')
parser.add_argument('-ipex', help='use intel extension for pytorch', action='store_true')
parser.add_argument('-bf16', help='use bfloat16', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def run(video_path, model_weight, jitter, vis, display_off, save_text, USE_CUDA=False, USE_IPEX=False, USE_BF16=False):
    total_start_time = time.time()
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(0)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    # set up output file
    if save_text:
        outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

    # set up face detection mode
    # if face_path is None:
    #     facemode = 'DLIB'
    # else:
    #     facemode = 'GIVEN'
    #     column_names = ['frame', 'left', 'top', 'right', 'bottom']
    #     df = pd.read_csv(face_path, names=column_names, index_col=0)
    #     df['left'] -= (df['right']-df['left'])*0.2
    #     df['right'] += (df['right']-df['left'])*0.2
    #     df['top'] -= (df['bottom']-df['top'])*0.1
    #     df['bottom'] += (df['bottom']-df['top'])*0.1
    #     df['left'] = df['left'].astype('int')
    #     df['top'] = df['top'].astype('int')
    #     df['right'] = df['right'].astype('int')
    #     df['bottom'] = df['bottom'].astype('int')

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        exit()

    # if facemode == 'DLIB':
    #     cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight,USE_CUDA=USE_CUDA)
    model_dict = model.state_dict()
    torch.set_num_threads(2*torch.get_num_threads())
    if USE_CUDA:
        snapshot = torch.load(model_weight)
    else:
        snapshot = torch.load(model_weight, map_location=torch.device('cpu'))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    if USE_CUDA:
        model.cuda()
    model.train(False)

    if USE_IPEX:
        if USE_BF16:
            model = ipex.optimize(model=model,dtype=torch.bfloat16)
        else:
            model = ipex.optimize(model=model)
    # video reading loop
    while(cap.isOpened()):
        frame_start_time = time.time()
        ret, frame = cap.read()
        print("readed")
        if ret == True:
            height, width, channels = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_cnt += 1
            bbox = get_face_loc(frame)
            frame = Image.fromarray(frame)
            if bbox is not []:
                for b in bbox:
                    face = frame.crop((b))
                    img = test_transforms(face)
                    img.unsqueeze_(0)
                    if jitter > 0:
                        for i in range(jitter):
                            bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                            bj = [bj_left, bj_top, bj_right, bj_bottom]
                            facej = frame.crop((bj))
                            img_jittered = test_transforms(facej)
                            img_jittered.unsqueeze_(0)
                            img = torch.cat([img, img_jittered])

                    # forward pass
                    if USE_BF16:
                        img = img.bfloat16()
                    if USE_CUDA:
                        output = model(img.cuda())
                    else:
                        output = model(img)
                    if jitter > 0:
                        output = torch.mean(output, 0)
                    score = F.sigmoid(output).item()

                    coloridx = min(int(round(score*10)),9)
                    draw = ImageDraw.Draw(frame)
                    drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                    draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
                    if save_text:
                        f.write("%d,%f\n"%(frame_cnt,score))

            if not display_off:
                frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('Result',frame)
                if cv2.waitKey()  == ord(' '):       # 每帧图像显示20ms，等待用户按下空格键退出
                    break
                print("showing")
                if vis:
                    outvid.write(frame)
        else:
            break
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_rate = 1/frame_time
        print("frame time:",frame_time,"\trealtime frame rate:",frame_rate)

    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    print('DONE!')
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print("total time:",total_time)


if __name__ == "__main__":
    run(args.video, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text, args.cuda, args.ipex, args.bf16)