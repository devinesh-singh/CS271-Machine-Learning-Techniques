#!/usr/bin/env python
# coding: utf-8

# In[161]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from math import sqrt
from scipy import stats
from sklearn.decomposition import PCA
import scipy.linalg as la
from scipy.spatial import distance


# In[178]:


# define the matrix:
B = np.array([[2, -2, -1, 3], [-1, 3, 3, -1], [0, 2, 3, 0], [1, 3, 1, 3], [1, 0, -1, 2], [-3, 2, 4, -1], [5, -1, 5, 3], [2, 1, 2, 0]])


# In[179]:


# compute the row means
row_means = np.mean(B, axis=1)


# In[180]:


# subtract the row means
B = B.astype(np.float)
for i in range(B.shape[1]): 
    B[:,i] -= row_means  


# In[181]:


# compute the C matrix
A = B
A_transpose = np.transpose(A)
C = (np.matmul(A, A_transpose))/4
print(C)


# In[182]:


# compute the eigenvalues/eigenvectors (use svd)
la.svd(C, full_matrices=True, compute_uv=True)


# In[188]:


# principal components
u1 = (np.array([[-0.46536222,   0.46992727,  0.29148827,  0.01799517, -0.22596605,
          0.59653918, -0.26182389,  0.04296201]]))
u2 = (np.array([[-0.03737162, -0.04179176, -0.1539934 ,  0.35316746,  0.19747961,
         -0.21056644, -0.84376578, -0.23051788 ]]))
u3 = (np.array([[0.48276203, -0.10894711,  0.03582972,  0.33194317,  0.22044514,
          0.61144594,  0.13821706, -0.44948322]]))


# In[189]:


X1 = A[:,0]
X1 = X1.reshape((8, 1))
X2 = A[:, 1]
X2 = X2.reshape((8, 1))
X3 = A[:,2]
X3 = X3.reshape((8, 1))
X4 = A[:, 3]
X4 = X4.reshape((8, 1))


# u1

# In[192]:


print(np.dot(u1, X1))
print(np.dot(u2, X1))
print(np.dot(u3, X1))

print(np.dot(u1, X2))
print(np.dot(u2, X2))
print(np.dot(u3, X2))

print(np.dot(u1, X3))
print(np.dot(u2, X3))
print(np.dot(u3, X3))

print(np.dot(u1, X4))
print(np.dot(u2, X4))
print(np.dot(u3, X4))

score_matrix = np.array([-1.106925, 1.279375, -2.680025, 2.507575, 1.54795, 0.54835, -1.20845,-0.88785])
score_matrix = score_matrix.reshape((2, 4))

score_matrix = np.array([[-4.71254981,
-1.15784725,
-1.52520957,
4.38421832,
3.26562107,
-0.69953493,
4.0654171,
-3.54380392,
0.53743636,
-3.73708561,
1.4360301,
1.68730814]])
score_matrix.reshape(3,4)


# In[193]:


# part B: initialize the vectors
Y1 = (np.array([[1, 5, 1, 5, 5, 1, 1, 3]]))
Y2 = (np.array([[-2, 3, 2, 3, 0, 2, -1, 1]]))
Y3 = (np.array([[2, -3, 2, 3, 0, 0, 2, -1]]))
Y4 = (np.array([[2, -2, 2, 2, -1, 1, 2, 2]]))


# In[200]:


# create a matrix with all y-vectors
Y = np.array([[1, 5, 1, 5, 5, 1, 1, 3], [-2, 3, 2, 3, 0, 2, -1, 1], [2, -3, 2, 3, 0, 0, 2, -1], [2, -2, 2, 2, -1, 1, 2, 2]])
Y.reshape(8,4)


# In[201]:


# subtract the mean for each y-vector
Y = Y.astype(np.float)
Y = np.transpose(Y)
for i in range(Y.shape[1]): 
    Y[:,i] -= row_means 


# In[203]:


a = (np.dot(u1, Y[:,0]))
b = (np.dot(u2, Y[:,0]))
c = (np.dot(u3, Y[:,0]))
d = (np.dot(u1, Y[:,1]))
e = (np.dot(u2, Y[:,1]))
f = (np.dot(u3, Y[:,1]))
g = (np.dot(u1, Y[:,2]))
h = (np.dot(u2, Y[:,2]))
i = (np.dot(u3, Y[:,2]))
j = (np.dot(u1, Y[:,3]))
k = (np.dot(u2, Y[:,3]))
l = (np.dot(u3, Y[:,3]))


# dist1 = np.sqrt(np.linalg.norm(a - score_matrix[0]))
# dist2 = np.linalg.norm(np.dot(Y1, u2) - score_matrix[1])
# dist3 = np.linalg.norm(np.dot(Y1, u3) - score_matrix[2])
# 
# print(dist1)
# print(dist2)
# print(dist3)
# 

# In[209]:


sum_sq = np.sum(np.square(a - score_matrix[:,0])) 
sum_sq


# In[210]:


sum_sq = np.sum(np.square(b - score_matrix[:,0])) 
sum_sq


# In[211]:


sum_sq = np.sum(np.square(c - score_matrix[:,0])) 
sum_sq


# In[212]:


sum_sq = np.sum(np.square(d - score_matrix[:,1])) 
sum_sq


# In[213]:


sum_sq = np.sum(np.square(e - score_matrix[:,1])) 
sum_sq


# In[214]:


sum_sq = np.sum(np.square(f - score_matrix[:,1])) 
sum_sq


# In[215]:


sum_sq = np.sum(np.square(g - score_matrix[:,2])) 
sum_sq


# In[216]:


sum_sq = np.sum(np.square(h - score_matrix[:,2])) 
sum_sq


# In[217]:


sum_sq = np.sum(np.square(i - score_matrix[:,2])) 
sum_sq


# In[218]:


sum_sq = np.sum(np.square(j - score_matrix[:,3])) 
sum_sq


# In[219]:


sum_sq = np.sum(np.square(k - score_matrix[:,3])) 
sum_sq


# In[220]:


sum_sq = np.sum(np.square(l - score_matrix[:,3])) 
sum_sq


# In[ ]:




