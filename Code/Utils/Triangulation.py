import numpy as np

def getPositiveZCount(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positiveZ = 0
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            n_positiveZ += 1

    return n_positiveZ

# def getX(line, y):
#     x = -(line[1]*y + line[2])/line[0]
#     return x
    
def ExtractCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C