import numpy as np


print("shape 1")
x = np.array([[1,2,3],[1,2,3]])
print(x.shape)


print("shape 2")
x = np.array([ [[1,2,3], [ 4, 5, 6]],
               [[7,8,9], [10,11,12]] ])
print(x.shape)

print(x[0,0,2])

