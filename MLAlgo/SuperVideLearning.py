import string
import iteratools
import functools
import statistics
import random

class EstimateTree:
    def __init__(self , est1: int , est2 : int):
        self.est1 = est1
        self.est2 = est2
        print(f"the value of the first estimator is {est1} and also the value of the second estimator is {est2}")
