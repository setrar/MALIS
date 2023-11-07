import numpy as np

def mean_squared_error(act, pred):
   return ((pred - act) ** 2).mean()

act = np.array([1.1,2,1.7])
pred = np.array([1,1.7,1.5])

print(mean_squared_error(act,pred))
