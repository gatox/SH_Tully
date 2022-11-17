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
import math 
import matplotlib.pyplot as plt
from scipy.linalg import eig, expm
import random 
from scipy.integrate import ode

# =============================================================================
# Functions
# =============================================================================


def position(x, v, a, dt):
	x_new = x + v*dt + 0.5*a*dt*dt
	return x_new


#def gradiente(dig_grad_jj, h_jj, h_ii, k_ij):
#	grad = dig_grad_jj - (h_ii - h_jj)*k_ij
#	return grad

#def aceleration(m, grad):
#	a = -(1/m)*grad
#	return a

def velocity(v, a_0, a_1, dt):
	v_new = v + 0.5*(a_0 + a_1)*dt
	return v_new


def propagator(h_ij, v, k_ij, dt):
    p_ij = expm(-(1j*(h_ij) + v*k_ij)*dt)
    return p_ij

def dz_dt(t, z, a11, a12, a21, a22):
    c_i, c_j = z
    return [-1j*a11*c_i - a12*c_j, -a21*c_i - 1j*a22*c_j]


def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, h_ij, v, k_ij, dt):
    hop_ji = (1 - ((np.abs(c_j_dt)**2)/(np.abs(c_j_t)**2)))* \
        (( (c_i_dt*(propagator(h_ij, v, k_ij, dt))[0,1]*(c_j_t)).real) / \
         ( (np.abs(c_j_t)**2) - \
          (c_j_dt*(propagator(h_ij, v, k_ij, dt))[1,1]*(c_j_t)).real))
    return hop_ji


def V_11(a, b, x):
    if x > 0:
        v = a*(1-math.exp(-b*x))
    else:
        v = -a*(1-math.exp(b*x))
    return v

def D_11(a, b, x):
    if x > 0:
        v = a*b*(math.exp(-b*x))
    else:
        v = a*b*(math.exp(b*x))
    return v


def V_12(c, d, x):
    v = c*math.exp(-d*(x**2))
    return v

def D_12(c, d, x):
    v = -2*d*c*x*math.exp(-d*(x**2))
    return v


def U_i(a,b,c,d,x):
    en_1 = -math.sqrt((V_11(a, b, x))**2 + (V_12(c, d, x))**2)
    return en_1

def Gra_U_i(a,b,c,d,x):
    gra_u_1 = (V_11(a,b,x)*D_11(a,b,x)+V_12(c,d,x)*D_12(c,d,x))/U_i(a,b,c,d,x)
    return gra_u_1

def U_j(a,b,c,d,x):
    en_2 = math.sqrt((V_11(a, b, x))**2 + (V_12(c, d, x))**2)
    return en_2

def Gra_U_j(a,b,c,d,x):
    gra_u_2 = (V_11(a,b,x)*D_11(a,b,x)+V_12(c,d,x)*D_12(c,d,x))/U_j(a,b,c,d,x)
    return gra_u_2

def nac_ij(a,b,c,d,x):
    nac = D_11(a,b,x)/(50*(U_j(a,b,c,d,x)-U_i(a,b,c,d,x)))  
    return nac

#def hopping_LZ(a,b,c,d,x,v):
#    hop_ji_LZ =math.exp(((-2*math.pi*(V_12(c,d,x))**2)/(2*D_11(a,b,x)))*v)
#    return hop_ji_LZ

# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH: Tully's case 1")
#time_step = input ("Enter the time step: " )
#time_step = float(time_step) #suggested time_step = 20, equivalent to 0.484 fs
#md_step = input ("Enter the number of MD steps: " )
#md_step = int(md_step)

time_step = float(1) #suggested time_step = 20, equivalent to 0.484 fs
md_step = int(100)

# =============================================================================
# Constants
# =============================================================================
m = 1       # mass atomic units
#k = 0.1       # srping constant

a = 0.01
b = 1.6
c = 0.005
d = 1.0

au1 = 0.0242  # 1 atomic unit (a.u.) of time is approximately 0.0242 fs
dt = au1*time_step
t_max = dt*md_step

# =============================================================================
# Initial conditions
# =============================================================================
#x = input ("Enter the initial position: " )
#x = float(x) #suggested x=0
#v = input ("Enter the initial velocity: " )
#v = float(v) #suggested v!=0
##ep = input ("Enter the couping value: " )
##ep = float(ep)
#c_i = input ("Coefficient of state i: " )
#c_i = float(c_i)
#c_j = input ("Coefficient of state j: " )
#c_j = float(c_j)

x = float(-3) #suggested x=0
v = float(2) #suggested v!=0
c_i = float(0)
c_j = float(1)



t = 0.0001
hop_ji = 0
hop_ji_LZ = 0
a_HO_t = 0
u_ij = np.array([U_i(a,b,c,d,x), U_j(a,b,c,d,x)])
vk = np.array([nac_ij(a,b,c,d,x), nac_ij(a,b,c,d,x)])
I = np.identity(len(u_ij))
H = np.zeros([len(u_ij),len(u_ij)])
I_rot = I[::-1]
rc = np.arange(-x, x, 0.1)

if c_j > 0 and c_i < 1:
    a_HO_t = -Gra_U_j(a,b,c,d,x)/m
else:
    a_HO_t = -Gra_U_i(a,b,c,d,x)/m


c_t = np.array([c_i, c_j])
norm_c_dt = c_t
time = []
pos = []
vel = []
cou = []
poten = []
popu = []
norm = []
hopp = []
#hopp_LZ = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
t0 = 0.0
z = ode(dz_dt).set_integrator('zvode', method='bdf')
z.set_initial_value(c_t, t0)


while(t <= t_max):
    pos.append(x)
    vel.append(v)
    time.append(t)
    poten.append(u_ij)
    popu.append(c_t)
    cou.append(vk)
    norm.append(norm_c_dt)
    hopp.append(hop_ji)
#    hopp_LZ.append(hop_ji_LZ)
    x = position(x, v, a_HO_t, dt)

    #Surface Hopping (Start)
    
    u_ij = np.array([U_i(a,b,c,d,x), U_j(a,b,c,d,x)])
    vk = np.array([nac_ij(a,b,c,d,x), nac_ij(a,b,c,d,x)])
    #c_dt = propagator(u_ij*I, v*vk, I_rot, dt).dot(c_t)
    z.set_f_params(u_ij[0],v*vk[0],v*vk[1],u_ij[1])
    c_dt = z.integrate(t)
    
    norm_c_dt = np.abs(c_dt)    
    hop_ji = hopping(c_dt[1], c_t[1], c_dt[0], c_t[0], u_ij*I, v*vk, I_rot, dt)
    hop_ji = (hop_ji > 0) * hop_ji #only positives
#    hop_ji_LZ = hopping_LZ(a,b,c,d,x,v)
    
    r = random.uniform(0, 1)
    
    if 0 < r <= hop_ji:
        a_HO_dt = -Gra_U_i(a,b,c,d,x)/m
        #print("Yes hopping at",t, "with r",r, "<=",hop_ji)
    else:
        a_HO_dt = -Gra_U_j(a,b,c,d,x)/m
        #print("No hopping at",t, "with r",r, "<=",hop_ji)
        
    #Surface Hopping (End)

    v = velocity(v, a_HO_t, a_HO_dt, dt)
    
    
    a_HO_t = a_HO_dt
    
    c_t = c_dt
    t = t + dt


# =============================================================================
# Plots position 
# =============================================================================

plt.plot(time,pos, label = 'x(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('X(t)', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots velocity
# =============================================================================


plt.plot(time,vel, label = 'v(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('V(t)', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots energies
# =============================================================================

plt.plot(time,np.array(poten)[:,1], label = 'E_1')
plt.plot(time,np.array(poten)[:,0], label = 'E_0')
plt.plot(time,np.array(cou)[:,0],linestyle='--', label = 'NAC')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
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

