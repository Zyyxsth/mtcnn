import cv2
from core.detect import create_mtcnn_net, MtcnnDetector
from core.vision import vis_face
import numpy as np

def cutimg(img, width, length, k=8):
    '''暂时仅支持k=8，期待后续改进'''
    i = 1
    while(i<=k):
        print(i)
        if(i<=k/2):
            small_img = img[0:int(width/2), int((i-1)*length*2/k):int(i*length*2/k)]
        else:
            small_img = img[int(width/2) : int(width), int((i-1-k/2)*length*2/k):int((i-k/2)*length*2/k)]

        if(i == k/2):
            small_img = np.power(small_img, 0.8).astype(np.uint8)

        bboxs, landmarks = mtcnn_detector.detect_face(small_img)

        if (i <= k/2):
            print("here")
            for x in bboxs:
                x[1] = x[1]
                x[0] = x[0] + int((i-1)*length*2/k)
                x[3] = x[3]
                x[2] = x[2] + int((i-1)*length*2/k)
            for y in landmarks:
                y[0] = y[0] + int((i-1)*length*2/k)
                y[2] = y[2] + int((i-1)*length*2/k)
                y[4] = y[4] + int((i-1)*length*2/k)
                y[6] = y[6] + int((i-1)*length*2/k)
                y[8] = y[8] + int((i-1)*length*2/k)
        else:
            print("there")
            for x in bboxs:
                x[1] = x[1] + int(width/2)
                x[0] = x[0] + int((i-1-k/2)*length*2/k)
                x[3] = x[3] + int(width/2)
                x[2] = x[2] + int((i-1-k/2)*length*2/k)
            for y in landmarks:
                y[1] = y[1] + int(width/2)
                y[3] = y[3] + int(width/2)
                y[5] = y[5] + int(width/2)
                y[7] = y[7] + int(width/2)
                y[9] = y[9] + int(width/2)
                y[0] = y[0] + int((i-1-k/2)*length*2/k)
                y[2] = y[2] + int((i-1-k/2)*length*2/k)
                y[4] = y[4] + int((i-1-k/2)*length*2/k)
                y[6] = y[6] + int((i-1-k/2)*length*2/k)
                y[8] = y[8] + int((i-1-k/2)*length*2/k)

        if (i == 1):
            newbboxs = bboxs
            newlandmarks = landmarks
            print("here")
        else:
            newbboxs = np.append(newbboxs, bboxs, axis=0)
            newlandmarks = np.append(newlandmarks, landmarks, axis=0)
            print("there")

        i = i + 1

    return newbboxs, newlandmarks

if __name__ == '__main__':
    # original model

    # p_model_path = "./original_model/pnet_epoch.pt" # 262
    p_model_path = "./model_store/pnet_batch8192_epoch_50.pt" # 235
    # p_model_path = "./model_store/pnet_batch4096_epoch_21.pt"  # 249

    # r_model_path = "./model_store/rnet_epoch_30.pt"
    r_model_path = "./original_model/rnet_epoch.pt"

    o_model_path = "./original_model/onet_epoch.pt "
    # o_model_path = "./model_store/onet_epoch_100.pt"

    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                        use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=3, threshold=[0.1, 0.1, 0.1])

    img = cv2.imread("mid_new.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    bboxs, landmarks = mtcnn_detector.detect_face(img)
    bboxs, landmarks = cutimg(img, 840, 1280, 8)
    bbox_num = bboxs.shape[0]
    print("bboxs : ", bbox_num)
    save_name = './test_result/' + str(bbox_num) + '_r_zchange.jpg'
    vis_face(img_bg, bboxs, landmarks, save_name)

    # #original model
    # o_model_path = "./original_model/onet_epoch.pt"

    # #trained model
    # p_model_path = "./model_store/pnet_epoch_11.pt"
    # r_model_path = "./model_store/rnet_epoch_9.pt"

    # pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path, use_cuda=False)
    # mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=12, threshold=[0.3, 0.1, 0.3])

    # img = cv2.imread("mid.jpg")
    # img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # bboxs, landmarks = mtcnn_detector.detect_face(img)

    # save_name = 'r_1.jpg'
    # vis_face(img_bg,bboxs,landmarks, save_name)
