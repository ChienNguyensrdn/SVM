from __future__ import print_function

import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist

# list of points 
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0.T, X1.T), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# plot points
plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize = 8, alpha = .8)
plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8)
plt.axis('equal')

# axis limits
plt.ylim(0, 3)
plt.xlim(2, 4)


# hide tikcs 
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlabel('$x_1$', fontsize = 20)
plt.ylabel('$x_2$', fontsize = 20)

plt.figure().savefig("svm.pdf")

# plt.show()


# find solution by sklearn
from sklearn.svm import SVC
clf = SVC(kernel = 'linear', C = 1e5) # just a big number 
y = y.reshape((2*N,))
clf.fit(X.T, y ) # each sample is one row

w = clf.coef_
b = clf.intercept_
print('w = ', w)
print('b = ', b)

# # OPTIONAL - VISUALIZATION
# with PdfPages('svm.pdf') as pdf:
#     # draw
#     # plot points
#     fig, ax = plt.subplots()

#     x1 = np.arange(-10, 10, 0.1)
#     print(w[0,1])
#     y1 = -w[0, 0]/w[0,1]*x1 - b/w[0,1]
#     y2 = -w[0, 0]/w[0,1]*x1 - (b-1)/w[0,1]
#     y3 = -w[0, 0]/w[0,1]*x1 - (b+1)/w[0,1]
#     plt.plot(x1, y1, 'k', linewidth = 3)
#     plt.plot(x1, y2, 'k')
#     plt.plot(x1, y3, 'k')


#     y4 = 10*x1
#     plt.plot(x1, y1, 'k')
#     plt.fill_between(x1, y1, color='red', alpha= 0.1)
#     plt.fill_between(x1, y1, y4, color = 'blue', alpha = 0.1)


#     plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize = 8, alpha = .8)
#     plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8)

#     plt.axis('equal')
#     plt.ylim(0, 3)
#     plt.xlim(2, 4)

#     # hide tikcs 
#     cur_axes = plt.gca()
#     cur_axes.axes.get_xaxis().set_ticks([])
#     cur_axes.axes.get_yaxis().set_ticks([])

#     # add circles around support vectors 
#     for m in S:
#         circle = plt.Circle((X[0, m], X[1, m] ), 0.1, color='k', fill = False)
#         ax.add_artist(circle)


#     plt.xlabel('$x_1$', fontsize = 20)
#     plt.ylabel('$x_2$', fontsize = 20)
# #     plt.savefig('svm.png', bbox_inches='tight', dpi = 300)
#     pdf.savefig()
#     plt.show()



