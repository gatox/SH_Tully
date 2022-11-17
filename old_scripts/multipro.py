#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 21:55:26 2022

@author: edisonsalazar
"""


from multiprocessing import Process


def f(l, i):
    


if __name__ == '__main__':
    
    initial_time = time.time()
    lock = Lock()
    for num in range(100):
        Process(target=f, args=(lock, num)).start()
    
    print(f"Time: {(time.time()-initial_time):>0.3f}s")
    
