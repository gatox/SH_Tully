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
import random 

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
    p_ij = expm(-(1j*(h_ij + v*k_ij))*dt)
    return p_ij


def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, h_ij, v, k_ij, dt):
    hop_ji = (1 - ((np.abs(c_j_dt)**2)/(np.abs(c_j_t)**2)))* \
        (( (c_i_dt*(propagator(h_ij, v, k_ij, dt))[0,1]*(c_j_t)).real) / \
         ( (np.abs(c_j_t)**2) - \
          (c_j_dt*(propagator(h_ij, v, k_ij, dt))[1,1]*(c_j_t)).real))
    return hop_ji


def U_i(k, x):
    en_1 = k*(x**2)
    return en_1

def Gra_U_i(k, x):
    gra_u_1 = -(k*x)
    return gra_u_1

def U_j(k, x):
    en_2 = k*(x-2)**2
    return en_2

def Gra_U_j(k, x):
    gra_u_2 = -(k*(x-2))
    return gra_u_2



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
c_i = input ("Coefficient of state i: " )
c_i = float(c_i)
c_j = input ("Coefficient of state j: " )
c_j = float(c_j)


t = 0
hop_ji = 0
a_HO_t = 0
u_ij = np.array([0.0, 0.0])
vk = np.array([ep, ep])
I = np.identity(len(u_ij))
H = np.zeros([len(u_ij),len(u_ij)])
I_rot = I[::-1]

if c_j > 0 and c_i < 1:
    a_HO_t = Gra_U_j(k, x)/m
else:
    a_HO_t = Gra_U_i(k, x)/m


c_t = np.array([c_i, c_j])
norm_c_dt = c_t
time = []
pos = []
vel = []
poten = []
popu = []
norm = []
hopp = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
while(t <= t_max):
    pos.append(x)
    vel.append(v)
    time.append(t)
    poten.append(u_ij)
    popu.append(c_t)
    norm.append(norm_c_dt)
    hopp.append(hop_ji)
    x = position(x, v, a_HO_t, dt)

    #Surface Hopping (Start)
    
    u_ij = np.array([U_i(k, x), U_j(k,x)])
    c_dt = propagator(u_ij*I, vk, I_rot, dt).dot(c_t)
    norm_c_dt = np.abs(c_dt)    
    hop_ji = hopping(c_dt[1], c_t[1], c_dt[0], c_t[0], u_ij*I, vk, I_rot, dt)
    hop_ji = (hop_ji > 0) * hop_ji #only positives
    r = random.uniform(0, 1)
    
    if 0 < r <= hop_ji:
        a_HO_dt = Gra_U_i(k, x)/m
        print("Yes hopping at",t, "with r",r)
    else:
        a_HO_dt = Gra_U_j(k, x)/m
        print("No hopping at",t, "with r",r)
        
    #Surface Hopping (End)

    v = velocity(v, a_HO_t, a_HO_dt, dt)
    a_HO_t = a_HO_dt
    
    c_t = c_dt
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

plt.plot(time,np.array(norm)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm)[:,0], label = '|C_i(t)|')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('|C_i| & |C_j|', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()
