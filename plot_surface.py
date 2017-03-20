import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])
xx, yy = np.meshgrid(x,y, sparse=True)
pairs = np.vstack([ xx.reshape(-1), yy.reshape(-1) ]).reshape(2,-1)
print(xx, yy)
