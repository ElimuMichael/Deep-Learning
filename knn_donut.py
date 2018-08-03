# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from util import get_donut
from knn import KNN

if __name__=='__main__':
   X, Y  = get_donut()
   
   plt.scatter(X[:,0], X[:, 1], s = 100, c = Y, alpha = 0.5)
   plt.show()
   
   for k in (1, 2, 3, 4, 5):
       model = KNN(k)
       model.fit(X, Y)
       print("Model with {} neighbor: - Accuracy: {}".format(k, model.score(X, Y)))
