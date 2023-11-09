from scipy.spatial import distance_matrix
from scipy.spatial import minkowski_distance as m_d
from collections import Counter
import numpy as np
from math import sqrt

"""
# https://machinelearningmastery.com/distance-measures-for-machine-learning/ 
# calculate minkowski distance
"""
#def minkowski_distance(a, b, p):
#return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

def minkowski_distance(a, b, p):
    return sum(distance(e1, e2, p) for e1, e2 in zip(a,b))

def distance(e1, e2, p):
    distance = (np.sum(abs(e1-e2)**p))**(1/p)
    return distance


class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''        
        self.X_train = X
        self.y_train = y

    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        predictions = [self._predict(x, p) for x in X_new]
        return predictions

    def _predict(self, x, p):
        # compute the distance
        distances = self.minkowski_dist(x, p)

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        y_hat = Counter(k_nearest_labels).most_common()
        return y_hat[0][0]
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        dst = minkowski_distance( X_new, self.X_train, p)
        
        return dst

"""
Testing: 
Compare minkowski_distance with scipy.spatial
"""
def main():
    # define data
    row1 = [10, 20, 15, 10, 5]
    row2 = [12, 24, 18, 8, 7]
    # calculate distance (p=1)
    print('manhattan')
    dist = minkowski_distance(row1, row2, 1); print(dist)
    dist = m_d(row1, row2, 1); print(dist)
    # calculate distance (p=2)
    print('euclidian')
    dist = minkowski_distance(row1, row2, 2); print(dist)
    dist = m_d(row1, row2, 2); print(dist)
    print('Object')
    k = KNN(1)
    k.train(row1,row2)
    dist = k.minkowski_dist(row2,1)

    print('Individual')
    # euclidian
    print(distance(12, 10, 2))
    # manhattan
    print(distance(13, 10, 1))

if __name__ == "__main__":
    main()
