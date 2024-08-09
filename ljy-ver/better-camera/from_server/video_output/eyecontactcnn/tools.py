import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize(image, faces):
    output = image.copy()
    for idx, face in enumerate(faces):
        coords = face[:-1].astype(np.int32)
        # Draw face bounding box
        cv2.rectangle(output, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
        # Draw landmarks
        cv2.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv2.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv2.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv2.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv2.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        # Put score
        cv2.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0))

    return output

def get_face_loc(image):
    '''
    读入图片，返回一个二维数组，表示所有人脸的坐标（左上点与右下点）
    '''
    score_threshold = 0.85
    nms_threshold = 0.3
    backend = cv2.dnn.DNN_BACKEND_DEFAULT
    target = cv2.dnn.DNN_TARGET_CPU
    
    # new_width = 1024  # 设置新的宽度
    # new_height = 768  # 设置新的高度
    # image = cv2.resize(image, (new_width, new_height))
    # Instantiate yunet
    yunet = cv2.FaceDetectorYN.create(
        model='./face_detection_yunet_2023mar.onnx',
        config='',
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=5000,
        backend_id=backend,
        target_id=target
    )

    yunet.setInputSize((image.shape[1], image.shape[0]))
    _, faces = yunet.detect(image)  # faces: None, or nx15 np.array
    # vis_image = visualize(image, faces)
    faces_rects=[]
    output = image.copy()
    for idx, face in enumerate(faces):
        coords = face[:-1].astype(np.int32)
        faces_rects.append([coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]])

    # vis = True
    # if vis:
    #     # cv2.namedWindow('xx', cv2.WINDOW_AUTOSIZE)
    #     print("\n\ntools:",len(faces_rects))
    #     cv2.imwrite('output.jpg', vis_image)
    #     cv2.waitKey(0)
    # Draw rectangles around the detected faces
    return faces_rects

if __name__ == '__main__':
    model = './face_detection_yunet_2023mar.onnx'
    image = cv2.imread('test-zyq.jpg')
    # image = cv2.imread('testfull.jpg')
    # print(get_face_loc(image))
    get_face_loc(image)
