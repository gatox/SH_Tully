#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:00:34 2020

@author: edisonsalazar

Trajectory Surface Hopping (Velocity-Verlet algorithm & Tully hopping)

"""

# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, expm

# =============================================================================
# Functions
# =============================================================================


def position(x, v, a, dt):
	x_new = x + v*dt + 0.5*a*dt*dt
	return x_new


def gradiente(dig_grad_jj, h_jj, h_ii, k_ij):
	grad = dig_grad_jj - (h_ii - h_jj)*k_ij
	return grad

def aceleration(m, grad):
	a = -(1/m)*grad
	return a

def velocity(v, a_0, a_1, dt):
	v_new = v + 0.5*(a_0 + a_1)*dt
	return v_new


def propagator(h_ij, v, k_ij, dt):
    p_ij = expm(-(1j*h_ij + v*k_ij)*dt)
    return p_ij


def coefficients(c_i, p_ij):
    c_j = p_ij*c_i
    return c_j

def hopping(c_j, c_i):
    hop_ij = (1 - ((c_j*c_j)/(c_i*c_i)))
    return hop_ij



def HO_aceleration(m, k, x):
	a = -(k*x)/m
	return a


def U_1(k, x):
    en_1 = k*(x**2)
    return en_1

def U_2(k, x):
    en_2 = k*(x-2)**2
    return en_2



# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH one paricle Velocity Verlect Algorithm")
time_step = input ("Enter the time step: " )
time_step = int(time_step) #suggested time_step = 20, equivalent to 0.484 fs
md_step = input ("Enter the number of MD steps: " )
md_step = int(md_step)

# =============================================================================
# Constants
# =============================================================================
m = 0.2       # mass 
k = 0.1       # srping constant
au1 = 0.0242  # 1 atomic unit (a.u.) of time is approximately 0.0242 fs
dt = au1*time_step 
t_max = dt*md_step

# =============================================================================
# Initial conditions
# =============================================================================
x = input ("Enter the initial position: " )
x = float(x) #suggested x=0
v = input ("Enter the initial velocity: " )
v = float(v) #suggested v!=0
ep = input ("Enter the couping value: " )
ep = float(ep) 
c_i = input ("Enter the coefficient i: " )
c_i = float(c_i) 
c_j = input ("Enter the coefficient j: " )
c_j = float(c_j) 


t = 0
u = np.array([0.0, 0.0])
vk = np.array([ep, ep])
I = np.identity(len(u))
H = np.zeros([len(u),len(u)])
I_rot = I[::-1]

a_HO_0 = HO_aceleration(m, k, x)
c_0 = np.array([c_i, c_j])
norm_c_1 = c_0
time = []
pos = []
vel = []
poten = []
popu = []
norm = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
while(t <= t_max): 
    pos.append(x)
    vel.append(v)
    time.append(t)
    poten.append(u)
    popu.append(c_0)  
    norm.append(norm_c_1)
    x = position(x, v, a_HO_0, dt)
    a_HO_1 = HO_aceleration(m, k, x)
    u = np.array([U_1(k, x), U_2(k,x)])
    c_1 = propagator(u*I, vk, I_rot, dt).dot(c_0)
    norm_c_1 = np.abs(c_1)
    v = velocity(v, a_HO_0, a_HO_1, dt)
    a_HO_0 = a_HO_1
    c_0 = c_1

    t = t + dt


# =============================================================================
# Plots position and velocity
# =============================================================================

plt.plot(time,pos, label = 'x(t)')
plt.plot(time,vel,linestyle='--', label = 'v(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('X(t) & V(t)', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots population 
# =============================================================================

plt.plot(time,np.array(norm)[:,1], label = 'C2(t)')
plt.plot(time,np.array(norm)[:,0], label = 'C1(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('C_1 & C_2', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()



    