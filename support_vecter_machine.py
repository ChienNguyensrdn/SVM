from os import X_OK
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
#svm/Social_Network_Ads.csv
N=150
datasets = pd.read_csv('svm/Social_Network_Ads.csv')
X = datasets.iloc[:, [2,3,4]].values
Y = datasets.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X0 =np.array([x for x in X_Train if x[2]==0])[:,[0,1]]#class 0
X1 =np.array([x for x in X_Train if x[2]==1])[:,[0,1]]#class 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X0 = sc_X.fit_transform(X0)
X1 = sc_X.fit_transform(X1)
am =np.array([x for x in X_Test if x[2]==0])[:,[0,1]]#class 0
duong=np.array([x for x in X_Test if x[2]==1])[:,[0,1]]#class 0

X_Test=np.array([x for x in X_Test ])[:,[0,1]]

X_Test = sc_X.transform(X_Test)


y = np.concatenate((np.ones((1, len(X0))), -1*np.ones((1, len(X1)))), axis = 1)
X = np.concatenate((X0.T, X1.T), axis = 1)


from cvxopt import matrix, solvers
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix((V.T.dot(V)))
K = matrix(K, K.size, 'd')

p = matrix(-np.ones((2*N, 1)))
# build A, b, G, h 

#If-----------------tuong ung C==0 or C is none--
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))

A = matrix(y) 
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
# Setting options:
# solvers.options['show_progress'] = True
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = \n', l.T)
epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]
# S=np.array([x for x in l ])[:,0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)

'''
StandardScaler thực hiện nhiệm vụ Tiêu chuẩn hóa . Thông thường một tập dữ liệu chứa các biến khác nhau về tỷ lệ. Ví dụ: một bộ dữ liệu nhân viên sẽ chứa cột AGE với các giá trị trên thang 20-70 và cột SALary với các giá trị trên thang 10000-80000 .
Vì hai cột này có quy mô khác nhau, chúng được Chuẩn hóa để có tỷ lệ chung trong khi xây dựng mô hình học máy.
'''
# X_Test = sc_X.transform(X_Test)
# import numpy as geek 
i=0
am=0
duong=0
for x in X_Test:
    flag =np.sign(w.T.dot(x)+b)
    if flag==[1.]:
        i=i+1
        # print(flag)
print(i)
