import numpy as np
import cv2
from Utils.ImageUtils import *
def reproject3DPoints(R, C, K, pts3D_4):

    k = 3
    I = np.identity(3)
    P = np.dot(K, np.dot(R[k], np.hstack((I, -C[k].reshape(3,1)))))

    X = pts3D_4[k]
    x_ = np.dot(P, X)
    x_ = x_/x_[2,:]

    x = x_[0, :].T
    y = x_[1, :].T
    return x, y


def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def get3DPoints(K1, K2, matched_pairs, R2, C2):
    pts3D_4 = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        pts3D = []
        x1 = matched_pairs[:,0:2].T
        x2 = matched_pairs[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)  
        pts3D_4.append(X)
    return pts3D_4

def getEpipolarLines(set1, set2, F, image0, image1, file_name, rectified = False):
    # set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = getX(line2, y2_min)
            x2_max = getX(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = getX(line1, y1_min)
            x1_max = getX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]



        cv2.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    image_1, image_2 = makeImageSizeSame([img_epi1, img_epi2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv2.resize(concat, (1920, 660))
    displaySaveImage(concat, file_name)
    # cv2.imshow("a", concat)
    # cv2.imwrite("epilines.png", concat)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return lines1, lines2


