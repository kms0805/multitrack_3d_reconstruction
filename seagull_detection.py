import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

## frame들의 평균을 통해서 배경 구하기
def meanOfFrames(video, start_frame, end_frame, xmin, xmax, ymin, ymax):
    print("getting backgorund")
    frame1 = utils.load_video_frame(video, start_frame)
    frame1 = frame1[ymin:ymax,xmin:xmax,:]
    sum = np.zeros(frame1.shape)
    n = end_frame - start_frame
    for i in range(n):
        if i%25 == 0:
            print(i,">>>>")
        frame = utils.load_video_frame(video, start_frame + i)
        frame = frame[ymin:ymax,xmin:xmax,:]
        sum = sum + frame
        mean = (sum/n).astype('uint8')
    print('finish')
    return mean

#background subtraction
def getMask(img,background,threshold):
    diff = cv2.absdiff(img,background)
    diff = diff.mean(axis=2)
    mask = diff > threshold
    mask = mask[:,:,np.newaxis]
    return mask

#색깔로 갈매기의 몸체 추출하기(흰색 부분만 추출)
def getBodyMaskByColor(img,threshold):
    if img.ndim == 3 :
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if img.ndim == 1:
        gray = img
    mask = gray > threshold
    mask = mask[:,:,np.newaxis]
    return mask

#mask에 해당하는 부분만 빨간색으로 칠해서 plot
def visualize_detection(img,mask,alpha = 0.5):
    if mask.ndim == 2:
        mask = mask[:,:,np.newaxis]
    if mask.dtype != 'bool':
        mask = mask.astype('bool')
    mask_inv = np.invert(mask)
    mask = mask.astype(np.uint8)
    mask_inv = mask_inv.astype(np.uint8)
    red = np.zeros(img.shape, dtype = 'uint8')
    red[:] = 255,0,0
    fg =  red * mask
    bg = img * mask_inv
    output = fg + bg
    plt.imshow(img)
    plt.imshow(output,alpha = alpha)


def getCentroid(frame, mean, plot = True):
    centroid = np.empty((0,2))
    kernel = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]],np.uint8)
    if plot:
        plt.imshow(frame)
    mask = getMask(frame, mean, 30)
    seagull = frame*mask.astype('uint8')

    body = getBodyMaskByColor(seagull,165)
    body = body.astype('uint8')
    """
    갈매기의 날개가 몸체를 반으로 나누는 경우가 있어서
    dilate를 통해 이러한 부분 보완
    """
    body = cv2.dilate(body, kernel, iterations = 1)

    #DBSCAN을 이용하여 개체별로 clustering
    nonzero_index = np.transpose(np.nonzero(body))
    features = nonzero_index
    clustering = DBSCAN(eps = 3, min_samples=6).fit(features)

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    unique_labels = set(labels)

    for i,k in enumerate(unique_labels):
        if k == -1: #노이즈
            break
        class_member_mask = (labels == k)
        xy = nonzero_index[class_member_mask & core_samples_mask]
        x = xy[:,1].mean()
        y = xy[:,0].mean()
        centroid = np.append(centroid,np.array([[x,y]]),axis=0)
        if plot:
            plt.plot(x, y, 'o', markerfacecolor='blue', markeredgecolor="None", markersize = 5, alpha = 1)
            plt.text(x, y, i)
    return centroid


if __name__ == '__main__':
    video1 = 'Trial6/Cam1_C0010.mp4'
    video2 = 'Trial6/Cam3_C0007.mp4'

    frame1 = utils.load_video_frame(video1,0)
    frame2 = utils.load_video_frame(video2,856)
    frame_num = 200 #detection 을 진행할 frame의 갯수

    xmin1, xmax1, ymin1 ,ymax1 = 1200, 2400, 1000, 1600
    frame1 = frame1[ymin1:ymax1,xmin1:xmax1,:]
    start_frame = 0
    end_frame = 100
    mean1 = meanOfFrames(video1, start_frame, end_frame, xmin1, xmax1, ymin1, ymax1)

    xmin2, xmax2, ymin2, ymax2 = 1600, 2800, 900, 1500
    frame2 = frame2[ymin2:ymax2,xmin2:xmax2,:]
    start_frame = 856
    end_frame = 956
    mean2 = meanOfFrames(video2, start_frame, end_frame, xmin2, xmax2, ymin2, ymax2)

    centroidInVideo1 = []
    centroidInVideo2 = []

    print("getting centroids")
    for i in range(frame_num):
        if i%25 == 0:
            print(i,">>>>")
        frame1 = utils.load_video_frame(video1,0+i)
        frame2 = utils.load_video_frame(video2,856+i)
        frame1 = frame1[ymin1:ymax1,xmin1:xmax1,:]
        frame2 = frame2[ymin2:ymax2,xmin2:xmax2,:]
        centroid1 = getCentroid(frame1,mean1,plot = False)
        centroid2 = getCentroid(frame2,mean2,plot = False)
        centroidInVideo1.append(centroid1)
        centroidInVideo2.append(centroid2)
    np.save('result/centroidInVideo1.npy',centroidInVideo1)
    np.save('result/centroidInVideo2.npy',centroidInVideo2)
