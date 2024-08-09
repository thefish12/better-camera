from is_sheltered import *
from is_focus import *
import cv2
def save_img(img):
    cv2.imwrite('./output.jpg',img)
def main(img,num_people,lim_focus=0.6,lim_shelter=0):
    # return
    
    '''
        读入图片,人数，返回（0/1/2/3）。0:可以拍摄，1:有遮挡，2:未注视摄像头，3:未注视摄像头且有遮挡
        bonus：返回不符合条件的人的位置
        依赖函数：get_focus_rate()，
                 get_sheltered_rate(),
                 get_face_loc() #可能需要
    '''
    save_img(img)
    tmp=get_focus_rate(img)
    focus=tmp[0]
    focus_rate=sum(focus)/num_people
    sheltered_rate=get_sheltered_rate(img,num_people)[0]
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

if '__main__'==__name__:
    path='./'
    file_name=('input.jpg')
    img=cv2.imread(path+file_name)
    print(main(img,1))
