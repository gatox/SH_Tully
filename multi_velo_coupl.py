#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:09:55 2021

@author: edisonsalazar
"""
# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, expm
import random 

# =============================================================================
# Functions
# =============================================================================
l = 3

v = np.array([[ 1,  0,  3],[ 3,  2, -1],[ 2, -1,  3],[ 2,  2, -3]])

k_01 = np.zeros([4,3])
k_12 = np.array([[1,2,1],[2,1,-1],[2,1,1],[2,1,-2]])
k_13 = np.array([[1,0,1],[-1,2,-1],[-2,0,-1],[1,2,2]])
k_23 = np.array([[2,2,-1],[0,-1,1],[1,2,-1],[0,1,2]])

k = np.array([ [k_01], [k_12], [k_13], [k_23]]) 

def d_matrix(v,k,l):
    D = np.zeros([l,l])
    for i in l:
        for j in l:
            D[i,j+1] = np.linalg.norm(v.dot(k[j+2].transpose()))
            D[j+1,i] = D[i,j+1]
    return D

