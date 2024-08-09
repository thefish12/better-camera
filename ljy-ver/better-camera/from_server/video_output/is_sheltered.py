from eyecontactcnn.tools import get_face_loc
import cv2
def get_sheltered_rate(img,num_people): 
    face_loc=get_face_loc(img)
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
    return (rate,all)

if "__main__"==__name__:
    img=cv2.imread("test.png")
    print(get_sheltered_rate(img,1))
