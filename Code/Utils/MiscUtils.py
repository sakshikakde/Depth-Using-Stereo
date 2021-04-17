import numpy as np

def siftFeatures2Array(sift_matches, kp1, kp2):
    matched_pairs = []
    for i, m1 in enumerate(sift_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    return matched_pairs