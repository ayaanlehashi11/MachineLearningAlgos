import string , random , statistics , itertools
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVM
from sklearn import datasets
from sklearn.metrics import classification_report , confusion_matrix

def data_classifier(x , y):
    svm = SVM()
#with binomial distribution we can calculate the probability of success or failure
def BinomialDistribution(x , p , n ):
    combination = itertools.combinations(n , x )
    upper_exp = n - x
    p1 = p ** x
    p2 = 1 - p
    binomial_distribution = combination * p1*(p2)**upper_exp
    return binomial_distribution



