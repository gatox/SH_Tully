import os
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import sys

from collections import namedtuple
from pysurf.database import PySurfDB

class Population:
    
    def __init__(self):
        self.fs = 0.02418884254

    def read_prop(self):
        prop = open("prop.inp", 'r+')    
        for line in prop:
            if "dt" in line:
                dt = int(line.split()[2])
            elif "mdsteps" in line:
                mdsteps = int(line.split()[2])
            elif "nstates" in line:
                nstates = int(line.split()[2])
            elif "states" in line:
                states = []
                for i in range(nstates):
                   states.append(int(line.split()[2+i]))
        sampling = open("sampling.inp", 'r+')    
        for line in sampling:
            if "n_conditions" in line:
                trajs = int(line.split()[2])
        properties = namedtuple("properties", "dt mdsteps nstates states trajs")
        return properties(dt, mdsteps, nstates, states, trajs)

    def read_db(self):
        acstate = []
        rootdir = 'prop'
        prop = self.read_prop()
        mdsteps = prop.mdsteps
        trajs = prop.trajs
        for rootdir, dirs, files in os.walk(rootdir):
            for subdir in dirs:
                path = os.path.join(rootdir, subdir)
                db = PySurfDB.load_database(os.path.join(path,"results.db"), read_only=True)
                acstate.append(np.array(db["currstate"]))
        popu = np.array(acstate)
        matrix  = np.empty([trajs,mdsteps + 1])*np.nan
        for traj, sta in enumerate(popu):
            for t in range(len(popu[traj][:])):
                matrix[traj,t] = popu[traj][t]
        return matrix

    def get_popu(self):
        prop = self.read_prop()
        dt = prop.dt
        states = prop.states
        popu  = self.read_db()  
        popu = popu.T
        mdsteps, trajs = popu.shape
        ave_time = []
        ave_popu = []
        for m in range(mdsteps):
            nans = 0
            ref = np.zeros(prop.nstates)
            for traj in range(trajs):
                if np.isnan(popu[m][traj]):
                    nans += 1
                else:
                    val = np.equal(popu[m][traj], states)
                    for i in range(len(states)):
                        if val[i]:
                            ref[i] += 1  
            ave_time.append(m*dt*self.fs)
            ave_popu.append(ref/int(trajs-nans))
        return ave_time, ave_popu

    #def pandas_popu(self):
    #    prop = self.read_prop()
    #    dt = prop.dt
    #    states = prop.states
    #    mdsteps = prop.mdsteps
    #    trajs = prop.trajs
    #    popu  = self.read_db()  
    #    popu = pd.DataFrame(popu)
    #    ave_time = []
    #    ave_popu = []
    #    for m in range(mdsteps):
    #        ave_time.append(m*dt*self.fs)
    #    for col in popu.columns:
    #        no_nans = len(popu[col]) - popu[col].isna().sum()
    #        val = np.equal(popu[col].values, states)
    #        for i in range(len(states)):
    #            if val[i]:
    #                val = popu[col].value_counts()/no_nans
    #        ave_popu.append(val.values)
    #    return ave_time, ave_popu

    def plot_population(self):
        prop = self.read_prop()
        nstates = prop.nstates
        time, population = self.get_popu()
        #time, population = self.pandas_popu()
        for i in range(nstates):
            plt.plot(time,np.array(population)[:,i], label = '$S_%i$' %i)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend()
        plt.savefig("population.pdf", bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    popu = Population()
    popu.plot_population()
