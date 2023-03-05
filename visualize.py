import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


parser = argparse.ArgumentParser(description='visualize')
parser.add_argument('--three_d', action='store_true')
parser.add_argument('--two_d', action='store_true')

args = parser.parse_args()
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

if (args.three_d):
    trajactory_3d_list = np.load('result/trajactory_3d_list.npy')
    frames = []
    colors = plt.cm.prism(np.linspace(0,1,100))
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_xlim(0,-150)
    ax.set_ylabel('y axis')
    ax.set_ylim(20,-130)
    ax.set_zlabel('z axis')
    ax.set_zlim(0,-150)
    ax.view_init(elev=70 ,azim=90)
    for i in range(len(trajactory_3d_list[0])):
        for j,t in enumerate(trajactory_3d_list):
            tripoints3d = t[i]

            if tripoints3d[-1] == 0:
                continue
                
            color = colors[j%100]
            
            ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2],marker='o', markerfacecolor=tuple(color), markeredgecolor=tuple(color), markersize = 1)
        canvas = FigureCanvas(fig)
        canvas.draw()
        now = np.array(canvas.renderer._renderer)
        now = cv2.cvtColor(now, cv2.COLOR_RGB2BGR)
        frames.append(now)
    utils.making_video('result/3d.mp4',frames,5)

if(args.two_d):
    trajactory_2d_each_list = np.load('result/trajactory_2d_each_list.npy')

    ## 2d video 두개 비교
    xmin1, xmax1, ymin1 ,ymax1 = 1200, 2400, 1000, 1600
    xmin2, xmax2, ymin2, ymax2 = 1600, 2800, 900, 1500

    frames = []
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

    for i in range(len(trajactory_2d_each_list [0])):

        full1 = utils.load_video_frame('Trial6/Cam1_C0010.mp4',i)
        full2 = utils.load_video_frame('Trial6/Cam3_C0007.mp4',i+856)
        frame1 = full1[ymin1:ymax1,xmin1:xmax1,:]
        frame2 = full2[ymin2:ymax2,xmin2:xmax2,:]

    
        for j,t in enumerate(trajactory_2d_each_list):
            traj = t[i]

            pts1 = traj[0:2]
            pts2 = traj[2:4]

            if pts1[0]<0 or pts2[0]<0:
                continue

            x1 = pts1[0].astype(np.int32)
            y1 = pts1[1].astype(np.int32)
            x2 = pts2[0].astype(np.int32)
            y2 = pts2[1].astype(np.int32)
            color = colours[j%11]
            cv2.circle(frame1,(x1,y1),6,color,1)
            cv2.putText(frame1,str(j), (x1-10,y1-20),0, 0.5, color,2)
            cv2.circle(frame2,(x2,y2),6,color,1)
            cv2.putText(frame2,str(j), (x2-10,y2-20),0, 0.5, color,2)
        frame = cv2.hconcat((frame1,frame2))
        frames.append(frame)

    utils.making_video('result/tracking2deach.mp4',frames,5)
            