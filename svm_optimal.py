
# Optimisation
import cvxopt
import cvxopt.solvers
# Math
import numpy as np
from numpy import linalg
#Scikit
from sklearn.svm import SVC
# Custom Code:
class SVM(object):

    def __init__(self, kernel='linear', C=0, gamma=1, degree=3):

        if C is None:
            C=0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        C = float(C)
        gamma = float(gamma)
        degree=int(degree)

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y,C=1, d=3):
        # Inputs:
        #   x   : vector of x data.
        #   y   : vector of y data.
        #   c   : is a constant
        #   d   : is the order of the polynomial.
        return (np.dot(x, y) + C) ** d

    def gaussian_kernel(self, x, y, gamma=0.5):
        return np.exp(-gamma*linalg.norm(x - y) ** 2 )

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):

                # Kernel trick.
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                if self.kernel=='gaussian':
                    K[i, j] = self.gaussian_kernel(X[i], X[j], self.gamma)   # Kernel trick.
                    self.C = None   # Not used in gaussian kernel.
                if self.kernel == 'polynomial':
                    K[i, j] = self.polynomial_kernel(X[i], X[j], self.C, self.degree)


        # Converting into cvxopt format:
        P = cvxopt.matrix(np.outer(y, y) * K)
        self._p=P
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        A = cvxopt.matrix(A, A.size, 'd')

        b = cvxopt.matrix(0.0)

        if self.C is None or self.C==0:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Restricting the optimisation with parameter C.
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Setting options:
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # Solve QP problem:
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.l = np.array(solution['x'])
      
        # Lagrange multipliers
        alphas = np.ravel(solution['x'])        # Flatten the matrix into a vector of all the Langrangian multipliers.

        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Bias (For linear it is the intercept):
        self.b = 0
        for n in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b = self.b / len(self.alphas)

        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.alphas)):
                self.w += self.alphas[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        # Create the decision boundary for the plots. Calculates the hypothesis.
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)

                    if self.kernel == 'linear':
                        s += a * sv_y * self.linear_kernel(X[i], sv)
                    if self.kernel=='gaussian':
                        s += a * sv_y * self.gaussian_kernel(X[i], sv, self.gamma)   # Kernel trick.
                        self.C = None   # Not used in gaussian kernel.
                    if self.kernel == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(X[i], sv, self.C, self.degree)

                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        # Hypothesis: sign(sum^S a * y * kernel + b).
        # NOTE: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
        return np.sign(self.project(X))

import matplotlib.pyplot as plt


    

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets = pd.read_csv('svm/Social_Network_Ads.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

i=0
for x in range(len(Y_Train)):
    if Y_Train[i]==0:
        Y_Train[i]=-1
    i=i+1
__svm =SVM(kernel='linear', C=5)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
__svm.fit(X_Train,Y_Train)
l = __svm.l
print('lambda = \n', l.T)
epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]
# S=np.array([x for x in l ])[:,0]

b=__svm.b
w=__svm.w
print(__svm.w)
print(__svm.b)
print(__svm.l)

# X_Test=np.array([x for x in X_Test ])[:,[0,1]]
y_predict = __svm.predict(X_Test)                                       # Fit the out-of-sample (predicted data).
print(y_predict)
i=0
for x in X_Test:
    flag =__svm.predict(x)
    if flag==1:
        i=i+1
        # print(flag)
print(i)
