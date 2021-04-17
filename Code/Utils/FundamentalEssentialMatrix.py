import numpy as np

def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def EstimateFundamentalMatrix(feature_matches):
    normalised = True

    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def errorF(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1tmp=np.array([x1[0], x1[1], 1]).T
    x2tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1tmp, np.dot(F, x2tmp))
    
    return np.abs(error)


def getInliers(features):
    n_iterations = 1000
    error_thresh = 0.02
    inliers_thresh = 0
    chosen_indices = []
    chosen_f = 0

    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = EstimateFundamentalMatrix(features_8)
        for j in range(n_rows):
            feature = features[j]
            error = errorF(feature, f_8)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_f = f_8

    filtered_features = features[chosen_indices, :]
    return chosen_f, filtered_features

def getEssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected