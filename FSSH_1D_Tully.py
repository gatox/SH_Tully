#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:00:34 2020

@author: edisonsalazar

Trajectory Surface Hopping (Velocity-Verlet algorithm & Fewest-Switches Surface Hopping)

"""

# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import random 

from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
from tully_model_3 import Tully_3
from tully_model_4 import Tully_4

# =============================================================================
# Functions
# =============================================================================


def position(x, v, a, dt):
	x_new = x + v*dt + 0.5*a*dt*dt
	return x_new


def velocity(v, a_0, a_1, dt):
	v_new = v + 0.5*(a_0 + a_1)*dt
	return v_new


def propagator(dia_ij, u_ij, uc_ij, dt):
    p_ij = np.linalg.multi_dot([u_ij, np.diag(np.exp( -1j * dia_ij * dt)), uc_ij])
    return p_ij


    #hopping of Tully
def hopping(state, rho, h_ij, dt):
    hop_ji = (2.0 * np.imag(rho[state,:] * h_ij[:,state] * dt))/(np.real(rho[state,state])) 
    hop_ji[state] = 0.0
    hop_ji = np.maximum(hop_ji, 0.0)
    return hop_ji



# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD FSSH: Tully's algorithm")



# =============================================================================
# Constants
# =============================================================================



"""Calling Tully's models:"""

Tully = Tully_1(a = 0.01, b = 1.6, c = 0.005, d = 1.0)
#Tully = Tully_2(a = 0.10, b = 0.28, c = 0.015, d = 0.06, e0 = 0.05)
#Tully = Tully_3(a = 0.0006, b = 0.10, c = 0.90)
#Tully = Tully_4(a = 0.0006, b = 0.10, c = 0.90, d = 4.0)

# =============================================================================
# Initial conditions
# =============================================================================


""" Initial position"""
x_0 = float(-5) 
""" Initial momentum """ 
p_0 = float(20) 
""" Mass atomic units """
m = 2000.0
""" Initial velocity """
v_0 = p_0 / m
""" Number of states """
nstates = int(2)
""" Ground state """
c_i = float(0)
""" Excited state """
c_j = float(1)

"""Tuning factor for the coupling strength in adiabatic basis """
F = 20.0

""" Adiabatic (1) or Diabatic (0)"""
representation = 1

"""Appiying adjutment of momentum
   yes = True;
   no = False
"""
momentum_change = "False"

"""Initial time"""
t = 0.0

dt = np.abs(0.05/v_0)
t_max = 2.0*np.abs(x_0/v_0)
md_step = int(t_max / dt)



""" Density matrix """
rho = np.zeros([ nstates, nstates ], dtype=np.complex128)

"""Choosing the initial state; GS or Exc.State and 
   computing the aceleration at t_0
"""

if c_j > 0 and c_i < 1:
    state = 1
    a_HO_t = -Tully._gradient(x_0)[state]/m
    
else:
    state = 0
    a_HO_t = -Tully._gradient(x_0)[state]/m
    
rho[1,1] = c_j
rho[0,0] = c_i


track_state = []
time = []
pos = []
vel = []
cou = []
poten = []
den = []
popu = []
norm_real = []
hopp = []
ran = []
hopp_2 = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
while(t <= t_max):
    track_state.append(state)
    """ Computing Hamiltonian and NAC at time t_0 
    """
    if representation == 1:
        u_ij = np.diag(Tully._energy(x_0)) #adiabatic representation at t_0
    else:
        u_ij = Tully._di_energy(x_0)       #diabatic representation at t_0
        F = 0.0
        
    vk = F*Tully._a_coupling(x_0)
    
    Ene, U = np.linalg.eigh(u_ij)
    
    H = (u_ij) - 1j*(v_0*vk)
    
    
    """ Computing Hamiltonian and NAC at time t_0 + dt 
    """
    
    x_dt = position(x_0, v_0, a_HO_t, dt)
    
    if representation == 1:
        u_ij_dt = np.diag(Tully._energy(x_dt)) #adiabatic representation at t_0 + dt
    else:
        u_ij_dt = Tully._di_energy(x_dt)       #diabatic representation at t_0 + dt
        F = 0.0

    vk_dt = F*Tully._a_coupling(x_dt)

    
    H_dt = (u_ij_dt) - 1j*(v_0*vk_dt)
    
    """ Averaged Hamiltonian (middle point between H(t) and H(t+dt))
          
        Computing propagation and coefficients:
            rho_ij(t) = c_i(t)*c_j(t)^{*}
            as c(t+dt) = U*exp(-j*D*dt)*U.T*c(t)
            then rho_ij(t+dt) = c_i(t+dt)*c_j(t+dt)^{*}
                              = U*exp(-j*D*dt)*U.T*c_i(t)*c_j(t)^{*}*U*exp(j*D*dt)*U.T
                              = U*exp(-j*D*dt)*U.T*rho_ij(t)*U*exp(j*D*dt)*U.T
    """
    
    H_av= 0.5*(H + H_dt)
    
    E_av_eig_dt, U_av_dt = np.linalg.eigh(H_av)
    
    UT_av_dt = U_av_dt.T.conj()
    

    Prop_dt = propagator(E_av_eig_dt, U_av_dt, UT_av_dt, dt)
    Prop_T_dt = Prop_dt.T.conj()
    
    rho_dt = np.linalg.multi_dot([Prop_dt, rho, Prop_T_dt])
    c_dt = np.diag(rho_dt)
    

    norm_c_dt_real = np.real(c_dt)
    
    """ Computing hopping from state_i -> state_j and the new aceleration. 
        Additionally, the momentum is adjusted
    """
    hop_ij = hopping(state, rho, H_av, dt)

    
    r = random.uniform(0, 1)
    acc_prob = np.cumsum(hop_ij)
    hops = np.less(r, acc_prob)
    diff_ji = E_av_eig_dt[1-state] -E_av_eig_dt[state]
    beta_ji = v_0*(0.5)*(vk[1-state,state] + vk_dt[1-state,state])
    alpha_ji = 0.5*( ((0.5)*(vk[1-state,state] + vk_dt[1-state,state]))**2 )/m
    
    if representation == 1:
        hops_2 = (((beta_ji)**2/(4*alpha_ji)) >= diff_ji)
    else:
        hops_2 = "True"
    
    if any(hops) and hops_2:
        state = 1 - state
        a_HO_dt = -Tully._gradient(x_dt)[state]/m
        
        """If the hopping is succeeded, then an adjustment 
           in the velocity is performed in order to preserve 
           the total energy 
        """
        if momentum_change  == "True":
            if (beta_ji < 0.0):
                gama_ji = (beta_ji + np.sqrt(beta_ji**2 + 4*alpha_ji*diff_ji))/(2*alpha_ji)
                v_0 = v_0 - gama_ji*(vk[1-state,state]/m)
            else:
                gama_ji = (beta_ji - np.sqrt(beta_ji**2 + 4*alpha_ji*diff_ji))/(2*alpha_ji)
                v_0 = v_0 - gama_ji*(vk[1-state,state]/m)
    else:
        state = state
        a_HO_dt = -Tully._gradient(x_dt)[state]/m
        
    """ Computing the new velocity and update the time 
    """
    
    v_dt = velocity(v_0, a_HO_t, a_HO_dt, dt)
    
    pos.append(x_0)
    vel.append(v_0)
    time.append(t)
    norm_real.append(norm_c_dt_real)
    hopp.append(acc_prob[1])
    ran.append(r)
    hopp_2.append(hops_2)
    poten.append(Ene)
    cou.append(vk[0,1])

    
    x_0 = x_dt
    v_0 = v_dt
    a_HO_t = a_HO_dt
    rho = rho_dt
    
    
    t = t + dt
    

# =============================================================================
# Plots position 
# =============================================================================

plt.plot(time,pos, label = 'x(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('X(t)', fontweight = 'bold', fontsize = 16)
#plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots velocity
# =============================================================================


plt.plot(time,vel, label = 'v(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('V(t)', fontweight = 'bold', fontsize = 16)
plt.legend()
plt.show()

# =============================================================================
# Plots energies
# =============================================================================

plt.plot(time,np.array(poten)[:,1], label = '$E_1$')
plt.plot(time,np.array(poten)[:,0], label = '$E_0$')
plt.plot(time,[cou/F for cou in cou],linestyle='--', label = 'NAC, F = %i' %F)
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('$\mathbf{E_0\ &\ E_1}$', fontsize = 16)
plt.legend()
plt.show()

# =============================================================================
# Plots population
# =============================================================================

plt.plot(time,np.array(norm_real)[:,1], label = '$|C_j(t)|^2$')
plt.plot(time,np.array(norm_real)[:,0], label = '$|C_i(t)|^2$')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('$\mathbf{|C_i|^2\ &\ |C_j|^2}$', fontsize = 16)
plt.legend()
plt.show()


# =============================================================================
# Plots energies vs position
# =============================================================================

plt.plot(pos,np.array(poten)[:,1], label = '$E_1$')
plt.plot(pos,np.array(poten)[:,0], label = '$E_0$')
plt.plot(pos,[cou/F for cou in cou],linestyle='--', label = 'NAC, F = %i' %F)
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('$\mathbf{E_0\ &\ E_1}$', fontsize = 16)
plt.legend()
plt.show()

# =============================================================================
# Printing md_step, time, hopping probability, coupling and current state
# =============================================================================
dash = '-' * 85
print(dash)
print ("{:>10s} {:>12s} {:>14s} {:>14s} {:>12s} {:>16s}".format("MD_steps", "Time", "Hoppping_P", "Random" , "NAC", "Track_State"))
print(dash)
for i in range(md_step):
    print ("{:10.0f} {:>12.2f} {:>14.6f} {:>14.6f} {:>12.6f} {:>16.0f}".format(i+1, time[i], hopp[i], ran[i], cou[i], track_state[i]))
