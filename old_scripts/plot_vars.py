#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 10:32:45 2021

@author: edisonsalazar
"""

import matplotlib.pyplot as plt
from pandas import read_csv

class PlotResults:
    
    def __init__(self, output):
        self.output = read_csv(output)
        
    def plot_position(self):
        plt.plot(self.output.Time,self.output.Position, label = 'x(t)')
        for i in range(self.output.shape[0]-1):
            if self.output.State[i] != self.output.State[i+1]:
                plt.axvline(x=self.output.Time[i+1],label='Hop. at %i'%self.output.Time[i+1],\
                            linestyle='--', c = 'purple') 
        plt.xlabel('Time (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{X(a0)}$', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_velocity(self):
        plt.plot(self.output.Time,self.output.Velocity, label = 'x(t)')
        for i in range(self.output.shape[0]-1):
            if self.output.State[i] != self.output.State[i+1]:
                plt.axvline(x=self.output.Time[i+1],label='Hop. at %i'%self.output.Time[i+1],\
                            linestyle='--', c = 'purple') 
        plt.xlabel('Time (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{V(a0/arb.u.)}$', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_energies(self):
        plt.plot(self.output.Time,self.output.Ene_0, label = '$E_0$')
        plt.plot(self.output.Time,self.output.Ene_1, label = '$E_1$')
        for i in range(self.output.shape[0]-1):
            if self.output.State[i] != self.output.State[i+1]:
                plt.axvline(x=self.output.Time[i+1],label='Hop. at %i'%self.output.Time[i+1],\
                            linestyle='--', c = 'purple')
        plt.xlabel('Time (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Energy(a.u.)}$', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_population(self):
        plt.plot(self.output.Time,self.output.Pop_0, label = '$|C_0(t)|^2$')
        plt.plot(self.output.Time,self.output.Pop_1, label = '$|C_1(t)|^2$')
        for i in range(self.output.shape[0]-1):
            if self.output.State[i] != self.output.State[i+1]:
                plt.axvline(x=self.output.Time[i+1],label='Hop. at %i'%self.output.Time[i+1],\
                            linestyle='--', c = 'purple')
        plt.xlabel('Time (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_refl(self):
        plt.plot(self.output.Momentum,self.output.Reflexion, label = 'Prob_refle')
        plt.xlabel('Momentum (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('Probability', fontweight = 'bold', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_lower(self):
        plt.plot(self.output.Momentum,self.output.Tran_Lower, label = 'Prob_lower')
        plt.xlabel('Momentum (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('Probability', fontweight = 'bold', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_upper(self):
        plt.plot(self.output.Momentum,self.output.Tran_Upper, label = 'Prob_upper')
        plt.xlabel('Momentum (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('Probability', fontweight = 'bold', fontsize = 16)
        plt.legend()
        return plt.show()
    
    def plot_ref_upper(self):
        plt.plot(self.output.Momentum,self.output.Ref_Upper, label = 'Prob_ref_upper')
        plt.xlabel('Momentum (arb.u.)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('Probability', fontweight = 'bold', fontsize = 16)
        plt.legend()
        return plt.show()