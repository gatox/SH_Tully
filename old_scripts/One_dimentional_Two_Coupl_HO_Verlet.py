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

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, pinv, expm
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


#def propagator(h_ij, v, k_ij, dt):
#    p_ij = expm( -1j*( (h_ij) - 1j*v*k_ij )*dt)
#    return p_ij
    
def propagator(h_ij, dt):
    p_ij = expm( -h_ij * dt)
    return p_ij

#def propagator(rho_ij, dia, U, UT, dt):
#    for i in range(len(rho_ij)):
#        for j in range(len(rho_ij)):
#            rho_ij[i,j] = rho_ij[i,j] * np.exp( -1j * ( dia[i] - dia[j] ) * dt)
#    p_ij = np.dot(U, np.dot(rho_ij[i,j], UT))
#    return p_ij


#def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, h_ij, dt):
#    hop_ji = (1 - ((np.abs(c_j_dt)**2)/(np.abs(c_j_t)**2)))* \
#        (( (c_i_dt*(propagator(h_ij, dt))[0,1]*(c_j_t)).real) / \
#         ( (np.abs(c_j_t)**2) - \
#          (c_j_dt*(propagator(h_ij, dt))[1,1]*(c_j_t)).real))
#    return hop_ji

def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, h_ij, dt):
    hop_ji = (1 - ((np.abs(c_j_dt)**2)/(np.abs(c_j_t)**2)))* \
        (( (c_i_dt*(propagator(h_ij, dt))[0,1]*(c_j_t)).real) / \
         ( (np.abs(c_j_t)**2) - \
          (c_j_dt*(propagator(h_ij, dt))[1,1]*(c_j_t)).real))
    return hop_ji


def U_i(k, x):
    en_1 = 0.1*k*(x**2)
    return en_1

def Gra_U_i(k, x):
    gra_u_1 = -0.2*(k*x)
    return gra_u_1

def U_j(k, x):
    en_2 = 0.1*k*((x-2)**2)
    return en_2

def Gra_U_j(k, x):
    gra_u_2 = -0.2*(k*(x-2))
    return gra_u_2



# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH one paricle Velocity Verlect Algorithm: Rabi oscillations")
#time_step = input ("Enter the time step: " )
#time_step = float(time_step) #suggested time_step = 20, equivalent to 0.484 fs
#md_step = input ("Enter the number of MD steps: " )
#md_step = int(md_step)

time_step = float(1) #suggested time_step = 20, equivalent to 0.484 fs
md_step = int(1500)

# =============================================================================
# Constants
# =============================================================================
m = 1       # mass
k = 1       # srping constant
au1 = 0.01  # 1 atomic unit (a.u.) of time is approximately 0.0242 fs
dt = au1*time_step
t_max = dt*md_step

# =============================================================================
# Initial conditions
# =============================================================================
#x = input ("Enter the initial position: " )
#x = float(x) #suggested x=0
#v = input ("Enter the initial velocity: " )
#v = float(v) #suggested v!=0
#ep = input ("Enter the couping value: " )
#ep = float(ep)
#c_i = input ("Coefficient of state i: " )
#c_i = float(c_i)
#c_j = input ("Coefficient of state j: " )
#c_j = float(c_j)


x = float(-10) #suggested x=0
v = float(3) #suggested v!=0
ep = float(5) # coupling
c_i = float(1)
c_j = float(0)


t = 0.0
hop_ji = 0
a_HO_t = 0
u_ij = np.array([U_i(k,x), U_j(k,x)])
vk = np.array([ep, ep])
I = np.identity(len(u_ij))
H = np.zeros([len(u_ij),len(u_ij)])
I_rot = I[::-1]
rc = np.arange(-x, x, 0.1)

#rho_t = np.zeros([len(u_ij),len(u_ij)], dtype=np.complex64)

if c_j > 0 and c_i < 1:
    a_HO_t = Gra_U_j(k, x)/m
    #rho_t[1,1] = c_j
else:
    a_HO_t = Gra_U_i(k, x)/m
    #rho_t[0,0] = c_i



c_t = np.array([c_i, c_j])
norm_c_dt_real = c_t
norm_c_dt_abs = c_t
#norm_c_dt = np.array([rho_t[0,0], rho_t[1,1]])
time = []
pos = []
vel = []
cou = []
poten = []
popu = []
norm_real = []
norm_abs = []
hopp = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
while(t <= t_max):
    pos.append(x)
    vel.append(v)
    time.append(t)
    poten.append(u_ij)
#    popu.append(c_t)
    cou.append(vk)
    norm_real.append(norm_c_dt_real)
    norm_abs.append(norm_c_dt_abs)
    hopp.append(hop_ji)
    
    #t = t_0
    u_ij = np.array([U_i(k, x), U_j(k,x)])
    
    H = 1j*(u_ij*I) + ((v*vk)*I_rot)
    
    E_eig_t, U_t = eig(H)
    
    UT_t = pinv(U_t)
    
    #t = t_0 + dt
    
    x_dt = position(x, v, a_HO_t, dt)

    #Surface Hopping (Start)
    
    #u_ij = np.array([U_i(k, x) -40*math.sin(ep*t), U_j(k,x, ep) -40*math.sin(ep*t)])
    
    
    
    u_ij_dt = np.array([U_i(k, x_dt), U_j(k,x_dt)])
    vk_dt = vk
    
    H_dt = 1j*(u_ij_dt*I) + ((v*vk_dt)*I_rot) 
    
    E_eig_dt, U_dt = eig(H_dt)
    
    UT_dt = pinv(U_dt)
    

    #rho_d_t = np.dot(UT_t, rho_t)
    c_dia_t = np.dot(UT_t, c_t)
    
    
    #c_dt = propagator(u_ij*I, v*vk, I_rot, dt).dot(c_t)
    #c_d_dt = np.dot(U_t, propagator(H_dt, dt).dot(c_d_t))
    
    #c_dia_dt = np.dot(UT_dt, propagator(H_dt, dt).dot(np.dot(U_t, c_dia_t)))
    c_dia_dt = np.dot(propagator(H_dt, dt), c_t)
    norm_c_dt_real = np.real(c_dia_dt)  
    norm_c_dt_abs = np.abs(c_dia_dt)
    
    hop_ji = hopping(c_dia_dt[1], c_dia_t[1], c_dia_dt[0], c_dia_t[0], H_dt, dt)
    hop_ji = (hop_ji > 0) * hop_ji #only positives
    r = random.uniform(0, 1)
    
    if 0 < r <= hop_ji:
        a_HO_dt = Gra_U_i(k, x_dt)/m
        print("Yes hopping at",t, "with r",r, "<=",hop_ji)
    else:
        a_HO_dt = Gra_U_j(k, x_dt)/m
        print("No hopping at",t, "with r",r, "<=",hop_ji)
        
    #Surface Hopping (End)
    
    x = x_dt

    v = velocity(v, a_HO_t, a_HO_dt, dt)
    a_HO_t = a_HO_dt
    
    c_dia_t = c_dia_dt
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
# Plots population real
# =============================================================================

plt.plot(time,np.array(norm_real)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm_real)[:,0], label = '|C_i(t)|')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('Real|C_i| & Real|C_j|', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots population abs
# =============================================================================

plt.plot(time,np.array(norm_abs)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm_abs)[:,0], label = '|C_i(t)|')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('ABS|C_i| & ABS|C_j|', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

## =============================================================================
## Plots population
## =============================================================================
#term=1000
#
#plt.plot(time[:term],np.array(norm)[:term,1], label = '|C_j(t)|')
#plt.plot(time[:term],np.array(norm)[:term,0], label = '|C_i(t)|')
#plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
#plt.ylabel('|C_i| & |C_j|', fontweight = 'bold', fontsize = 16)
#plt.grid(True)
#plt.legend()
#plt.show()

# =============================================================================
# Plots energies vs position
# =============================================================================

plt.plot(pos,np.array(poten)[:,1], label = 'E_1')
plt.plot(pos,np.array(poten)[:,0], label = 'E_0')
plt.plot(pos,np.array(cou)[:,0],linestyle='--', label = 'NAC')
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots energies
# =============================================================================

plt.plot(time,np.array(poten)[:,1], label = 'E_1')
plt.plot(time,np.array(poten)[:,0], label = 'E_0')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()
