import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.optimize import main
import utils
import itertools
import sort
import reconstruction

def delete_tail_in_traj(traj):
    for i,t in enumerate(traj):
        k = t.shape[0]-1
        for j in range(k,-1,-1):
            if t[j][4] == 1:
                traj[i] = traj[i][0:j+1,:]
                break
    return traj

def compare2traj(traj1,traj2,F,time_overlap_th,hits_th):
    """
    두 궤도 사이의 epipolar distance를 계산
    """
    start_time1 = traj1[0][3]
    end_time1 = traj1[-1][3]
    start_time2 = traj2[0][3]
    end_time2 = traj2[-1][3]
    start_time = np.max((start_time1,start_time2))
    end_time = np.min((end_time1,end_time2))
    time_len = np.int32(end_time - start_time + 1)
    cost_sum = 0
    hits1 = 0
    hits2 = 0
    if time_len < 1 :
        return 100000 
    
    for i in range(time_len):
        
        
        time1 = np.int32(start_time - start_time1 + i)
        time2 = np.int32(start_time - start_time2 + i)
        
        hits1 += traj1[time1][4] 
        hits2 += traj2[time2][4]

        index1 = traj1[time1][1:3]
        index2 = traj2[time2][1:3]
        
        lines1 = cv2.computeCorrespondEpilines(index1.reshape(-1,1,2), 1,F)
        lines2 = cv2.computeCorrespondEpilines(index2.reshape(-1,1,2), 2,F)
        
        hom1 = reconstruction.cart2hom(index1)
        hom2 = reconstruction.cart2hom(index2)

        cost_sum += np.abs(np.dot(lines1,hom2)) + np.abs(np.dot(lines2,hom1))
       
    if time_len < time_overlap_th:
        cost_sum = 1000000
    if hits1 < hits_th or hits2 < hits_th:
        cost_sum = 1000000
    
    return (cost_sum/time_len)

def is_overlapped(traj1,traj2):
    start_time1 = traj1[0][3]
    end_time1 = traj1[-1][3]
    start_time2 = traj2[0][3]
    end_time2 = traj2[-1][3]
    start_time = np.max((start_time1,start_time2))
    end_time = np.min((end_time1,end_time2))
    time_len = np.int32(end_time - start_time + 1)
    if time_len > 0:
        return True
    return False

def is_overlapped_between_2matches(match1,match2,traj1,traj2):
    result = False
    
    for m1t1 in match1[0]:
        for m2t1 in match2[0]:
            if is_overlapped(traj1[m1t1],traj1[m2t1]):
                result = True
    
    for m1t2 in match1[1]:
        for m2t2 in match2[1]:
            if is_overlapped(traj2[m1t2],traj2[m2t2]):
                result = True
                
    return result

class MultiViewTrajactory(object):
    def __init__(self,traj_lists,max_time, num_view):
        self.max_time = max_time
        self.num_view = num_view
        self.matched_traj_list = []
        self.traj2d_lists = []
        self.costmatrix_list = []
        self.unmatches_list = [] #list of trajactory's index
        self.matches = []        #list of trajactory's index  

        for i in range(num_view):
            self.traj2d_lists.append(traj_lists[i])
            self.unmatches_list.append(np.array(range(len(self.traj2d_lists[i]))))

        combi = itertools.combinations(range(num_view),2) #n개의 view 중 2개를 골라서 비교
        for i in combi:
            self.costmatrix_list.append(np.zeros((len(traj_lists[i[0]]),len(traj_lists[i[1]])),dtype=np.float32))
        
    def update_costmatrix(self,F,time_overlap_th,hits_th):
        traj_lists = []
        for i in range(self.num_view):
            traj_lists.append(self.traj2d_lists[i])
        combi = itertools.combinations(range(self.num_view),2) #n개의 view 중 2개를 골라서 비교
        for k, c in enumerate(combi):
            for i,t1 in enumerate(traj_lists[c[0]]):
                for j,t2 in enumerate(traj_lists[c[1]]):
                    self.costmatrix_list[k][i,j] = compare2traj(t1, t2, F, time_overlap_th,hits_th)
    def match(self, threshold):
        '''
        unmatches1,2 -> np array of num
        matches -> list of tuple(num_list1,num_list2)
        '''
        unmatches1 = self.unmatches_list[0]
        unmatches2 = self.unmatches_list[1]
        costmatrix = self.costmatrix_list[0]
        matches  = self.matches
        traj1 = self.traj2d_lists[0]
        traj2 = self.traj2d_lists[1]
        while(1):    
            if len(unmatches1) == 0 or len(unmatches2) == 0:
                break
                
            min_idx = np.unravel_index(np.argmin(costmatrix), costmatrix.shape)
            min_cost = costmatrix[min_idx]
            if min_cost > threshold:
                break
            t1 = min_idx[0]
            t2 = min_idx[1]
            
            costmatrix[t1,t2] = 44444 
            
            ##################################################### case1
            if t1 in unmatches1 and t2 in unmatches2:
                unmatches1 = np.delete(unmatches1,np.where(unmatches1 == t1))
                unmatches2 = np.delete(unmatches2,np.where(unmatches2 == t2))
                matches.append([[t1],[t2]])
            ##################################################### case2
            elif t1 in unmatches1 and t2 not in unmatches2:
                
                is_found = False # 안겹치는 match를 찾았는지 체크
                
                for i,m in enumerate(matches):
                    if t2 in m[1]:
                        is_found = True
                        for mt in m[0]:
                            if is_overlapped(traj1[t1],traj1[mt]):
                                is_found = False
                                break
                        if is_found:
                            m[0].append(t1)
                            unmatches1 = np.delete(unmatches1,np.where(unmatches1 == t1))
                            break
                            
            ##################################################### case3
            elif t2 in unmatches2 and t1 not in unmatches1:
                
                is_found = False # 안겹치는 match를 찾았는지 체크
                
                for i,m in enumerate(matches):
                    if t1 in m[0]:
                        is_found = True
                        for mt in m[1]:
                            if is_overlapped(traj2[t2],traj2[mt]):
                                is_found = False
                                break
                        if is_found:
                            m[1].append(t2)
                            unmatches2 = np.delete(unmatches2,np.where(unmatches2 == t2))
                            break
                            
            ################################################# case4
            elif t1 not in unmatches1 and t2 not in unmatches2:
                
                is_in_same_already = False
                t1_belong_index = 0
                t2_belong_index = 0
                for i,m in enumerate(matches):
                    if t1 in m[0] and t2 in m[1]:
                        is_in_same_already = True
                        break
                    if t1 in m[0]:
                        t1_belong_index = i
                    if t2 in m[1]:
                        t2_belong_index = i
                
                if is_in_same_already:
                    continue
                if is_overlapped_between_2matches(matches[t1_belong_index],matches[t2_belong_index],traj1,traj2):
                    continue
                    
                new1 = matches[t1_belong_index][0] + matches[t2_belong_index][0]
                new2 = matches[t1_belong_index][1] + matches[t2_belong_index][1]
                matches[t1_belong_index] = [new1,new2]
                matches.pop(t2_belong_index)


        self.unmatches_list[0] = unmatches1
        self.unmatches_list[1] = unmatches2
        self.costmatrix_list[0] = costmatrix
        self.matches = matches

    def update_matched_traj(self):
        self.matched_traj_list = []
        for m in self.matches:
            matched_traj = -1*np.ones((self.max_time,4),dtype = np.float32)
            for i in m[0]:
                st = self.traj2d_lists[0][i][0][3].astype(np.int32)
                et = self.traj2d_lists[0][i][-1][3].astype(np.int32)
                matched_traj[st:et+1,0:2] = self.traj2d_lists[0][i][:,1:3].copy()
            for i in m[1]:
                st = self.traj2d_lists[1][i][0][3].astype(np.int32)
                et = self.traj2d_lists[1][i][-1][3].astype(np.int32)
                matched_traj[st:et+1,2:4] = self.traj2d_lists[1][i][:,1:3].copy()
            self.matched_traj_list.append(matched_traj)


if __name__ == '__main__':
    #np load error(?)
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    centroidInVideo1 = np.load('result/centroidInVideo1.npy')
    centroidInVideo2 = np.load('result/centroidInVideo2.npy')
    F = np.load('result/F.npy')

    tracker1 = sort.Sort()
    tracker2 = sort.Sort()
    frame_num = len(centroidInVideo1)

    #### SORT Tracking
    print(">>> Sort Tracking")
    for i in range(frame_num):
        centroid1 = centroidInVideo1[i]
        centroid2 = centroidInVideo2[i]
        trackers1 = tracker1.update(centroid1)
        trackers2 = tracker2.update(centroid2)

    traj1 = []
    traj2 = []

    for trk1 in tracker1.trackers:
        traj1.append(trk1.get_trajactory())
    for trk1 in tracker1.deleted_trackers:
        traj1.append(trk1.get_trajactory())
    for trk2 in tracker2.trackers:
        traj2.append(trk2.get_trajactory())
    for trk2 in tracker2.deleted_trackers:
        traj2.append(trk2.get_trajactory())

    traj1 = delete_tail_in_traj(traj1)
    traj2 = delete_tail_in_traj(traj2)
    ####

    #### Matching
    print(">>> Matching")
    mvt = MultiViewTrajactory([traj1,traj2],max_time = frame_num, num_view = 2)
    mvt.update_costmatrix(F,20,10)
    mvt.match(2)
    mvt.update_matched_traj()
    mvt.update_costmatrix(F,10,5)
    mvt.match(2)
    mvt.update_matched_traj()
    ###

    #### result saving
    print(">>> result saving")
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = reconstruction.compute_P_from_fundamental(F)
    traj_num = len(mvt.matched_traj_list)
    trajactory_3d_list = np.zeros((traj_num ,frame_num, 4))
    trajactory_2d_each_list = np.zeros((traj_num ,frame_num, 4))
    for i in range(frame_num):
        for j,matched_traj in enumerate(mvt.matched_traj_list):
            pts1 = matched_traj[i,0:2]
            pts2 = matched_traj[i,2:4]
            trajactory_2d_each_list[j][i][:] = matched_traj[i,:]
            if pts1[0]<0 or pts2[0]<0:
                continue
            hom1 = reconstruction.cart2hom(pts1).reshape(1,-1)
            hom2 =  reconstruction.cart2hom(pts2).reshape(1,-1)
            tripoints3d =  reconstruction.linear_triangulation(hom1.T, hom2.T, P1, P2).reshape(1,4)
            trajactory_3d_list[j][i][:] = tripoints3d

    np.save('result/trajactory_3d_list.npy', trajactory_3d_list)
    np.save('result/trajactory_2d_each_list.npy', trajactory_2d_each_list)
    print('finish')
    ####
