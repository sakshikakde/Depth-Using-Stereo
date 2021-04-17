import numpy as np

def SSD(mat1, mat2):
    diff_sq = np.square(mat1 - mat2)
    ssd = np.sum(diff_sq)
    return ssd

def SAD(mat1, mat2):
    return np.sum(abs(mat1 - mat1))

def NCC(patch1, patch2):
  patch1_hat, patch2_hat = patch1 - patch1.mean(), patch2 - patch2.mean()
  upper = np.sum(patch1_hat*patch2_hat)
  lower = np.sqrt(np.sum(patch1_hat*patch1_hat)*np.sum(patch2_hat*patch2_hat))
  return upper/lower
