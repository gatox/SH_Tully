import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
import csv

from time import sleep
from tqdm.auto import tqdm
from pandas import (read_csv, DataFrame)
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
from collections import (namedtuple, Counter)
from pysurf.database import PySurfDB

class Population:
    
    def __init__(self, skip):
        self.skip = skip
        self.ev = 27.211324570273 
        self.fs = 0.02418884254
        self.aa = 0.5291772105638411 
        self.fs_rcParams = '20'
        self.t_0 = 0
        self.t_max = 200 #fs
        self.fs_ylabel = 18
        self.fs_xlabel = 18
        self.fs_yticks = 18
        self.fs_xticks = 18
        self.legend = "yes"
        self.label = "fssh"
        self.err = 3
        self.ci = [121.5,90]
        #self.ci = [106.5,71.7]

    def skip_traj(self):
        if self.skip == "yes":
            trajs = []
            read = open("trajectories_ok_list", 'r+') 
            return [line.split("/")[1] for line in read]
        return None

    def dis_dimer(self, m, a, b):
        return (np.sqrt(np.sum((np.array(m[int(a)])-np.array(m[int(b)]))**2)))*self.aa

    def dihedral(self, m, a, b, c, d):
        vec_a = np.array(m[int(a)])
        vec_b = np.array(m[int(b)])
        vec_c = np.array(m[int(c)])
        vec_d = np.array(m[int(d)])
        return calc_dihedral(Vector(vec_a),Vector(vec_b),Vector(vec_c),Vector(vec_d))* 180 / np.pi 

    def angle(self, m, a, b, c):
        vec_a = np.array(m[int(a)])
        vec_b = np.array(m[int(b)])
        vec_c = np.array(m[int(c)])
        return calc_angle(Vector(vec_a),Vector(vec_b),Vector(vec_c))* 180 / np.pi 

    def pyramidalization_angle(self, m, a, b, c, o):
        vec_a = np.array(m[int(a)]) - np.array(m[int(o)])
        vec_b = np.array(m[int(b)]) - np.array(m[int(o)])
        vec_c = np.array(m[int(c)]) - np.array(m[int(o)])
        vec_u = np.cross(vec_a, vec_b)
        d_cu = np.dot(vec_c,vec_u)
        cr_cu = np.cross(vec_c, vec_u)
        n_cr_cu = np.linalg.norm(cr_cu)
        angle = np.math.atan2(n_cr_cu,d_cu)
        return 90 - np.degrees(angle) 

    def monoexponetial_S1(self, t, t_d, t_e):
        return np.exp(-(t -t_d)/t_e)

    def monoexponetial_S0(self, t, t_d, t_e):
        return 1 -np.exp(-(t -t_d)/t_e)

    def confidence_interval_95(self, y, para, pcov):
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
        n = len(y)    # number of data points
        p = len(para) # number of parameters
        dof = max(0, n - p) # number of degrees of freedom
        # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof)
        results = {}
        for i, p,var in zip(range(n), para, np.diag(pcov)):
            sigma = var**0.5
            results.update({i:[p,sigma*tval]})
            #print(f"p{i} {p:>0.3f} +/- {sigma*tval:>0.3f}")
        return results

    def read_prop(self):
        sampling = open("sampling.inp", 'r+')    
        for line in sampling:
            if "n_conditions" in line:
                trajs = int(line.split()[2])
        traj_allowed = self.skip_traj()
        if  self.skip == "yes":
            trajs = len(traj_allowed)
        prop = open("prop.inp", 'r+')    
        tully = None
        for line in prop:
            if "dt" in line:
                tully = "yes"
                dt = int(float(line.split()[2]))
            elif "mdsteps" in line:
                mdsteps = int(line.split()[2])
            elif "nstates" in line:
                nstates = int(line.split()[2])
            #LZ
            elif "time_final" in line:
                time_final = int(line.split()[3])
            elif "timestep" in line:
                timestep = float(line.split()[3])
            elif "n_states" in line:
                nstates = int(line.split()[2])
            #LZ
            if "states" in line and not ("nstates" or "nghost_states") in line and tully is not None:
                states = [int(line.split()[2+i]) for i in range(nstates)]
            elif "method =" in line:
                method = str(line.split()[2])
                if method == "Surface_Hopping":
                    self.results = "results.db"
                    properties = namedtuple("properties", "dt mdsteps nstates states trajs")
                    return properties(dt, mdsteps, nstates, states, trajs)
                elif method == "LandauZener":
                    self.results = "prop.db"
                    properties = namedtuple("properties", "dt mdsteps nstates states trajs")
                    return properties(timestep/self.fs, int(time_final/timestep), nstates, [i for i in range(nstates)], trajs)

    def read_db(self):
        acstate = []
        crd = []
        rootdir = 'prop'
        prop = self.read_prop()
        mdsteps = prop.mdsteps
        trajs = prop.trajs
        matrix_0  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_1  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_2  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_3  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_4  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_5  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_6  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_7  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_8  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_9  = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_10 = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_11 = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_12 = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_13 = np.empty([trajs,mdsteps + 1])*np.nan
        matrix_14 = np.empty([trajs,mdsteps + 1])*np.nan
        traj = 0
        allowed = self.skip_traj()
        for rootdir, dirs, files in os.walk(rootdir):
            for subdir in dirs:
                if  self.skip == "yes" and subdir not in allowed: 
                    continue
                path = os.path.join(rootdir, subdir)
                print("Reading database from:",path)
                db = PySurfDB.load_database(os.path.join(path,self.results), read_only=True)
                row = len(np.array(db["currstate"]))
                #row = len(np.array(db["fosc"]))
                for t in range(row):
                    pop = np.array(db["currstate"][t])
                    var = np.array(db["fosc"][t], dtype=float)
                    matrix_0[traj][t] = var[0]
                    matrix_1[traj][t] = var[1]
                    etot = np.array(db["etot"][t])
                    ene = np.array(db["energy"][t])
                    if t==0:
                        self.ini_etot = etot[0] 
                    matrix_2[traj][t] = pop[0]
                    matrix_3[traj][t] = self.dihedral(np.array(db["crd"][t], dtype=float),3,0,1,4) #H4-C1-N2-H5
                    matrix_4[traj][t] = self.dihedral(np.array(db["crd"][t], dtype=float),2,0,1,4) #H3-C1-N2-H5
                    matrix_5[traj][t] = (etot[0]-self.ini_etot)*self.ev
                    matrix_6[traj][t] = self.dis_dimer(np.array(db["crd"][t], dtype=float),0,1) #C1-N2  
                    matrix_7[traj][t] = self.dis_dimer(np.array(db["crd"][t], dtype=float),1,4) #N2-H5  
                    matrix_8[traj][t] = self.dis_dimer(np.array(db["crd"][t], dtype=float),0,3) #C1-H4  
                    matrix_9[traj][t] = self.dis_dimer(np.array(db["crd"][t], dtype=float),0,2) #C1-H3  
                    matrix_10[traj][t] = self.angle(np.array(db["crd"][t], dtype=float),0,1,4) #C1-N2-H5  
                    matrix_11[traj][t] = self.pyramidalization_angle(np.array(db["crd"][t], dtype=float),3,2,1,0) #H4-H3-N2-C1  
                    matrix_12[traj][t] = (ene[1]-ene[0])*self.ev #Energy gap between S_1 and S_0 
                    matrix_13[traj][t] = ene[0]
                    matrix_14[traj][t] = (ene[1]+ene[0])/2 #Average between (S_1 and S_0)
                matrix_14[traj][:]=(matrix_14[traj][:]-matrix_13[traj][:].min())*self.ev # - E_S_0_min
                traj +=1
        #        #acstate.append(np.array(db["currstate"]))
        #        acstate.append(np.array(db["fosc"]))
        #matrix_1  = np.empty([trajs,mdsteps + 1])*np.nan
        #for traj, sta in enumerate(acstate):
        #    for t in range(len(acstate[traj])):
        #        matrix_1[traj,t] = np.array(acstate)[traj][t]
        #var = namedtuple("var", "p_c0 p_c1") 
        #return var(matrix_0,matrix_1)
        #var = namedtuple("var", "p_c0 p_c1 pop") 
        #return var(matrix_0,matrix_1,matrix_2)
        var = namedtuple("var", "p_c0 p_c1 pop dihe_3014 dihe_2014 etot dis_r12 dis_r25 dis_r14 dis_r13 angle_014 pyr_3210 e_gap ave") 
        return var(matrix_0, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9, matrix_10, matrix_11, matrix_12, matrix_14)

    def plot_population_compare(self, time, popu):
        prop = self.read_prop()
        nstates = prop.nstates
        fig, ax = plt.subplots()
        plt.rcParams['font.size'] = self.fs_rcParams 
        plt.plot(time,np.array(popu)[:,0], label = ' ', color='lime')
        plt.plot(time,np.array(popu)[:,1], label = ' ', color='red')
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=self.fs_xticks)
        plt.yticks(fontsize=self.fs_yticks)
        plt.ylim([0, 1])
        plt.xlabel('Time / fs', fontsize = 20)
        plt.ylabel('Population', fontsize = 20)
        plt.legend().remove()
        ax.text(0.05, 0.15, "$S_0$", transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='lime', va='top')
        ax.text(0.05, 0.95, "$S_1$", transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='red', va='top')
        ax.spines['right'].set_visible(True)
        plt.savefig("population_compare.pdf", bbox_inches='tight')
        plt.savefig("population_compare.png", bbox_inches='tight')
        plt.close()

    def plot_population_reactive_no_reactive_S0(self, time, popu, popu_S0_cis, popu_S0_trans):
        prop = self.read_prop()
        nstates = prop.nstates
        fig, ax = plt.subplots()
        for i in range(nstates):
            plt.plot(time,np.array(popu)[:,i], label = '$S_%i$' %i)
        plt.plot(time,popu_S0_cis, label = '$S_0$_cis', color='darkgreen',linestyle='-.')
        plt.plot(time,popu_S0_trans, label = '$S_0$_trans', color='darkred',linestyle='--')
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        #plt.legend(loc='center right',fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 15}, ncol=4)
        ax.spines['right'].set_visible(True)
        ax1 = ax.twinx()
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_reactive_no_reactive_S0.pdf", bbox_inches='tight')
        plt.savefig("population_reactive_no_reactive_S0.png", bbox_inches='tight')
        plt.close()

    def get_qy_popu(self):
        prop = self.read_prop()
        states = prop.states
        trajs = prop.trajs
        dt = prop.dt
        nstates = prop.nstates
        mdsteps = prop.mdsteps
        ave_time = []
        ave_popu = []
        ave_popu_S0_cis= []
        ave_popu_S0_trans = []
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        popu = pop.to_numpy()[:,1:] # removing time column
        dihe_2014 = self.no_time("dihe_2014.dat")
        analysis_qy = open("analysis_qy.out", "w")
        analysis_qy.write(f"MD_steps Time(fs) Trajs Pop_S0(in %) Pop_S1(in %) QY_S0_cis(in %) QY_S0_trans(in %)\n")
        for m in range(mdsteps+1): #time_steps
            nans = 0
            ref = np.zeros(nstates)
            ref_S0_trans = 0
            ref_S0_cis = 0
            for traj in range(trajs):
                if np.isnan(popu[m][traj]):
                    nans += 1
                else:
                    val = np.equal(popu[m][traj], states)
                    for i in range(len(states)):
                        if val[i]:
                            ref[i] += 1
                            if (abs(np.array(dihe_2014)[0,traj]-np.array(dihe_2014)[m,traj])>=100 and i == 0):
                                ini = np.array(dihe_2014)[0,traj]
                                sec = np.array(dihe_2014)[m,traj]
                                ref_S0_trans += 1
                                #if m == 827 and abs(ini-sec)>170:
                                #    print("Traj_>=100:",ini, sec, abs(ini-sec), traj, m)
                            elif(abs(np.array(dihe_2014)[0,traj]-np.array(dihe_2014)[m,traj])<100 and i == 0):
                                ini = np.array(dihe_2014)[0,traj]
                                sec = np.array(dihe_2014)[m,traj]
                                ref_S0_cis += 1
                                #if m == 827 and abs(ini-sec)<15:
                                #    print("Traj_<100:",ini, sec, abs(ini-sec), traj, m)
            if int(trajs-nans) != 0:
                analysis_qy.write(f"{int(m):>4.0f} {m*dt*self.fs:>9.2f} {trajs-nans:>9.0f} {ref[0]:>7.0f}({(ref[0]/int(trajs-nans))*100:>0.1f}) {ref[1]:>7.0f}({(ref[1]/int(trajs-nans))*100:>0.1f}) {ref_S0_cis:>5.0f}({(ref_S0_cis/int(trajs-nans))*100:>1.1f}) {ref_S0_trans:>5.0f}({(ref_S0_trans/int(trajs-nans))*100:>1.1f})\n")
                ave_time.append(m*dt*self.fs)
                ave_popu.append(ref/int(trajs-nans))
                ave_popu_S0_cis.append(ref_S0_cis/int(trajs-nans))
                ave_popu_S0_trans.append(ref_S0_trans/int(trajs-nans))
            else:
                break
        analysis_qy.close()
        self.plot_population_reactive_no_reactive_S0(ave_time, ave_popu, ave_popu_S0_cis, ave_popu_S0_trans)
        self.plot_population_compare(ave_time, ave_popu)
        return ave_time, ave_popu

    def _get_popu_c(self, filename,trajs):
        ave_time = []
        ave_popu = []
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                ref = 0
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val == 'nan':
                        nans += 1
                    else:
                        ref += float(val)
                if int(trajs-nans) == 0:
                    break
                else:
                    ave_popu.append(ref/int(trajs-nans))
        return ave_time, ave_popu
        

    def get_popu_c(self):
        prop = self.read_prop()
        trajs = prop.trajs
        ave_time_0 = []
        ave_time_1 = []
        ref_c0 = [] 
        ref_c1 = [] 
        ave_time_0, ref_c0 = self._get_popu_c("p_c0.dat",trajs)
        ave_time_1, ref_c1 = self._get_popu_c("p_c1.dat",trajs)
        return ave_time_0, [ref_c0,ref_c1]

    def get_popu_adi(self):
        prop = self.read_prop()
        states = prop.states
        trajs = prop.trajs
        nstates = prop.nstates
        ave_time = []
        ave_popu = []
        filename = "pop.dat"
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                ref = np.zeros(nstates)
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val == 'nan':
                        nans += 1
                    else:
                        state = np.equal(float(val), states)
                        for i in range(len(states)):
                            if state[i]:
                                ref[i] += 1  
                if int(trajs-nans) == 0:
                    break
                else:
                    ave_popu.append(ref/int(trajs-nans))
        return ave_time, ave_popu

    def get_popu(self, var_0, var_1):
        prop = self.read_prop()
        dt = prop.dt
        states = prop.states
        popu_0 = var_0.T
        popu_1 = var_1.T
        mdsteps, trajs = popu_0.shape
        ave_time = []
        ave_popu = []
        results_dia = open("results_dia.out", "w")
        head = namedtuple("head","time diabatic_1 diabatic_0")
        head = head("Time", "Dia_1", "Dia_0")
        results_dia.write(f"{head.time},{head.diabatic_1},{head.diabatic_0}\n")
        for m in range(mdsteps):
            nans = 0
            ref = np.zeros(prop.nstates)
            for traj in range(trajs):
                if np.isnan(popu_0[m][traj]):
                    nans += 1
                else:
                    #val = np.equal(popu[m][traj], states)
                    #for i in range(len(states)):
                    #    if val[i]:
                    #        ref[i] += 1  
                    ref[0] +=  popu_0[m][traj]
                    ref[1] +=  popu_1[m][traj]
            if int(trajs-nans) == 0:
                break
            else:
                #print(int(m),int(m*dt*self.fs),int(trajs-nans))#number of live trajectories 
                ave_time.append(m*dt*self.fs)
                ave_popu.append(ref/int(trajs-nans))
                var = namedtuple("var", "time dia_1 dia_0")
                var = var(m*dt*self.fs, ref[1]/int(trajs-nans),ref[0]/int(trajs-nans))
                results_dia.write(f"{var.time:>0.7f},{var.dia_1:>0.7f},{var.dia_0:>0.7f}\n")
        results_dia.close()
        return ave_time, ave_popu

    def get_var_ave(self, var):
        prop = self.read_prop()
        dt = prop.dt
        var = var.T
        mdsteps, trajs = var.shape
        ave_time = []
        ave_var = []
        for m in range(mdsteps):
            nans = 0
            ref = 0
            for traj in range(trajs):
                if np.isnan(var[m][traj]):
                    nans += 1
                else:
                    ref += var[m][traj]
            if int(trajs-nans) != 0:
                #print(int(m),int(m*dt*self.fs),int(trajs-nans))#number of live trajectories 
                ave_time.append(m*dt*self.fs)
                ave_var.append(ref/int(trajs-nans))
            else:
                break
        return ave_time, ave_var

    def get_all_var(self, var):
        prop = self.read_prop()
        dt = prop.dt
        var = var.T
        mdsteps, trajs = var.shape
        print(mdsteps, trajs)
        time = [m*dt*self.fs for m in range(mdsteps)]
        return time, var

    #def get_all_var(self):
    #    prop = self.read_prop()
    #    traj = prop.trajs
    #    mdsteps = prop.mdsteps
    #    dis = read_csv("dis_r12.dat") 
    #    pop = read_csv("pop.dat") 
    #    #pop = pop.transpose()
    #    pop = pop.to_numpy()[:,1:]
    #    time = dis['time']
    #    time = time.to_numpy()
    #    com, row = pop.shape
    #    #for i,j in enumerate(time):
    #    #    print(i,j)
    #    #for k in range(1,traj):
    #    #    print(k)
    #    return pop[:,99]

    def plot_population_adi(self):
        prop = self.read_prop()
        nstates = prop.nstates
        time, population = self.get_popu_adi()
        fig, ax = plt.subplots()
        for i in range(nstates):
            plt.plot(time,np.array(population)[:,i], label = '$S_%i$' %i)
        #plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        #plt.legend(loc='center right',fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), prop={'size': 15}, ncol=2)
        ax.spines['right'].set_visible(True)
        ax1 = ax.twinx()
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_adi.pdf", bbox_inches='tight')
        plt.savefig("population_adi.png", bbox_inches='tight')
        plt.close()

    def plot_population_pop_c(self):
        prop = self.read_prop()
        nstates = prop.nstates
        time, population = self.get_popu_adi()
        time_c, population_c = self.get_popu_c()
        fig, ax = plt.subplots()
        plt.plot(time,np.array(population)[:,0], color='#1f77b4', label = '$S_0$ CP')
        plt.plot(time,np.array(population)[:,1], color='#ff7f0e', label = '$S_1$ CP')
        plt.plot(time,population_c[0], color='#1f77b4', linestyle='--', label = '$S_0$ QP')
        plt.plot(time,population_c[1], color='#ff7f0e', linestyle='--', label = '$S_1$ QP')
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        #plt.legend(loc='center right',fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), prop={'size': 15}, ncol=4)
        ax.spines['right'].set_visible(True)
        ax1 = ax.twinx()
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_curr_cvec.pdf", bbox_inches='tight')
        plt.savefig("population_curr_cvec.png", bbox_inches='tight')
        plt.close()

    def plot_population_adi_fitted(self):
        prop = self.read_prop()
        nstates = prop.nstates
        time, population = self.get_popu_adi()
        params_S0, cv_S0 = curve_fit(self.monoexponetial_S0, time, np.array(population)[:,0])
        params_S1, cv_S1 = curve_fit(self.monoexponetial_S1, time, np.array(population)[:,1])
        S0_t_d = params_S0[0]
        S0_t_e = params_S0[1]
        S1_t_d = params_S1[0]
        S1_t_e = params_S1[1]
        for i in range(nstates):
            plt.plot(time,np.array(population)[:,i], label = '$S_%i$' %i)
        plt.plot(time, self.monoexponetial_S0(time, S0_t_d, S0_t_e), '--', label="fitted S0")
        plt.plot(time, self.monoexponetial_S1(time, S1_t_d, S1_t_e), '--', label="fitted S1")
        plt.xlim([self.t_0, self.t_max])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend(loc='center right',fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("population_adi_fitted.pdf", bbox_inches='tight')
        plt.savefig("population_adi_fitted.png", bbox_inches='tight')
        plt.close()
        print(f"Time constants t_d and t_e of the S1")
        self.conf_95_S1 = self.confidence_interval_95(np.array(population)[:,1], params_S1, cv_S1)
        print(int(round(self.conf_95_S1[0][0],0)), int(round(self.conf_95_S1[1][0],0)), int(round(self.conf_95_S1[0][0],0)+ round(self.conf_95_S1[1][0],0)))
        print(f"\nTime constants t_d and t_e of the S0")
        self.conf_95_S0 = self.confidence_interval_95(np.array(population)[:,0], params_S0, cv_S0)
        print(self.conf_95_S0)

    def plot_population_dia(self):
        prop = self.read_prop()
        nstates = prop.nstates
        var  = self.read_db()  
        time, population = self.get_popu(var.pop_0, var.pop_1)
        for i in range(nstates):
            plt.plot(time,np.array(population)[:,i], label = '$S_%i$' %i)
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("population_dia.pdf", bbox_inches='tight')
        plt.close()

    def compare(self, val, comp, ep):
        lower = np.abs(val-ep) 
        upper = np.abs(val+ep)
        myBool = True
        if comp > lower and comp < upper:
            return myBool
        return not myBool

    def con_int_two_angles(self, angle_1, angle_2):
        angle_alpha = read_csv(angle_1)       
        angle_phi = read_csv(angle_2)         
        angle_alpha = angle_alpha.to_numpy()[:,1:] # removing time column
        angle_phi = angle_phi.to_numpy()[:,1:] # removing time column
        return angle_alpha, angle_phi

    def no_time(self, data):
        res = read_csv(data)       
        res = res.to_numpy()[:,1:] # removing time column
        return res 

    def plot_dihedral_hops_time(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        dihe_name = "dihe_3014.dat"
        pop_name = "pop.dat"
        dihedral = read_csv(dihe_name) 
        pop = read_csv(pop_name)
        hop = pop.to_numpy()[:,1:] # removing time column
        time = dihedral['time']
        time = time.to_numpy()
        dihedral = dihedral.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                x = time[j]
                y_dihe_1 = dihedral[j,i]
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([x,y_dihe_1])
                    else:
                        hop_10 = plt.scatter(x,y_dihe_1,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([x,y_dihe_1])
                    else:
                        hop_01 = plt.scatter(x,y_dihe_1,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([self.t_0, self.t_max])
        plt.ylim([-180, 180])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+dihe_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+dihe_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        plt.close()

    def plot_angle_hops_time(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        dihe_name = "dihe_3014.dat"
        angle_name = "angle_014.dat"
        pop_name = "pop.dat"
        dihedral = read_csv(dihe_name) 
        pop = read_csv(pop_name)
        hop = pop.to_numpy()[:,1:] # removing time column
        time = dihedral['time']
        time = time.to_numpy()
        dihedral = dihedral.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                x = time[j]
                y_dihe_1 = dihedral[j,i]
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([x,alpha])
                    else:
                        hop_10 = plt.scatter(x,alpha,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([x,alpha])
                    else:
                        hop_01 = plt.scatter(x,alpha,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([self.t_0, self.t_max])
        plt.ylim([0, 180])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+angle_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+angle_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        plt.close()

    def plot_pyra_hops_time(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        pyr_name = "pyr_3210.dat"
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                x = time[j]
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([x,pyram])
                    else:
                        hop_10 = plt.scatter(x,pyram,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([x,pyram])
                    else:
                        hop_01 = plt.scatter(x,pyram,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([self.t_0, self.t_max])
        plt.ylim([-180, 180])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+pyr_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+pyr_name.replace(".dat","")+"_hops_time.pdf", bbox_inches='tight')
        plt.close()

    def plot_angle_dihedral_hops(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        dihe_name = "dihe_2014.dat"
        angle_name = "angle_014.dat"
        pop_name = "pop.dat"
        dihedral = read_csv(dihe_name) 
        pop = read_csv(pop_name)
        hop = pop.to_numpy()[:,1:] # removing time column
        time = dihedral['time']
        time = time.to_numpy()
        dihedral = dihedral.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_3014.dat")
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                x = time[j]
                y_dihe_1 = dihedral[j,i]
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([alpha,y_dihe_1])
                    else:
                        hop_10 = plt.scatter(alpha,y_dihe_1,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([alpha,y_dihe_1])
                    else:
                        hop_01 = plt.scatter(alpha,y_dihe_1,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.ylim([-180, 180])
        plt.xlim([0, 180])
        plt.xlabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = self.fs_ylabel)
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+angle_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+angle_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
        plt.close()

    def plot_pyra_angle_hops(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        pyr_name = "pyr_3210.dat"
        dihe_name = "dihe_2014.dat"
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([phi,pyram])
                    else:
                        hop_10 = plt.scatter(phi,pyram,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([phi,pyram])
                    else:
                        hop_01 = plt.scatter(phi,pyram,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize = self.fs_ylabel)
        plt.xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize = self.fs_xlabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+pyr_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+pyr_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
        plt.close()

    def plot_ene_angle_hops(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        ave_name = "ave.dat"
        ene_name = "e_gap.dat"
        pyr_name = "pyr_3210.dat"
        dihe_name = "dihe_2014.dat"
        pop_name = "pop.dat"
        alpha_name = "angle_014.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        ene = self.no_time(ene_name) 
        ene_ave = self.no_time(ave_name) 
        mecp_10 = []
        mecp_01 = []
        hop_10 = []
        hop_01 = []
        mecp_t_10 = []
        mecp_t_01 = []
        hop_t_10 = []
        hop_t_01 = []
        mecp_ave_10 = []
        mecp_ave_01 = []
        hop_ave_10 = []
        hop_ave_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                t = time[j]
                ene_d = ene[j,i]
                ene_a = ene_ave[j,i]
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([alpha,ene_d])
                        mecp_t_10.append([t,ene_d])
                        mecp_ave_10.append([t,ene_a])
                    else:
                        hop_10.append([alpha,ene_d])
                        hop_t_10.append([t,ene_d])
                        hop_ave_10.append([t,ene_a])
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([alpha,ene_d])
                        mecp_t_01.append([t,ene_d])
                        mecp_ave_01.append([t,ene_a])
                    else:
                        hop_01.append([alpha,ene_d])
                        hop_t_01.append([t,ene_d])
                        hop_ave_01.append([t,ene_a])
        #self._ene_angle_hops(hop_10, hop_01, mecp_10, mecp_01, ene_name, alpha_name)
        #self._ene_time_hops(hop_t_10, hop_t_01, mecp_t_10, mecp_t_01, ene_name)
        self._ave_ene_time_hops(hop_ave_10, hop_ave_01, mecp_ave_10, mecp_ave_01, ave_name)

    def _ave_ene_time_hops(self, hop_ave_10, hop_ave_01, mecp_ave_10, mecp_ave_01, ave_name):
        plt.rcParams['font.size'] = self.fs_rcParams
        for i in hop_ave_10:
            hop_10 = plt.scatter(i[0],i[1],color='green', marker='o',s=35)
        for i in hop_ave_01:
            hop_01 = plt.scatter(i[0],i[1],color='red', marker='o',s=35)
        for i in mecp_ave_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_ave_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([0, 200])
        plt.ylim([0, 7])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('Average Energy (eV)', fontweight = 'bold', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+ave_name.replace(".dat","_")+"time_hops.pdf", bbox_inches='tight')
            plt.savefig("fssh_"+ave_name.replace(".dat","_")+"time_hops.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+ave_name.replace(".dat","_")+"time_hops.pdf", bbox_inches='tight')
            plt.savefig("lz_"+ave_name.replace(".dat","_")+"time_hops.png", bbox_inches='tight')
        plt.close()

    def _ene_time_hops(self, hop_t_10, hop_t_01, mecp_t_10, mecp_t_01, ene_name):
        plt.rcParams['font.size'] = self.fs_rcParams
        for i in hop_t_10:
            hop_10 = plt.scatter(i[0],i[1],color='green', marker='o',s=35)
        for i in hop_t_01:
            hop_01 = plt.scatter(i[0],i[1],color='red', marker='o',s=35)
        for i in mecp_t_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_t_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([0, 200])
        plt.ylim([0, 3])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{E_1 - E_0 (eV)}$', fontsize = self.fs_ylabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+"time_hops.pdf", bbox_inches='tight')
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+"time_hops.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+ene_name.replace(".dat","_")+"time_hops.pdf", bbox_inches='tight')
            plt.savefig("lz_"+ene_name.replace(".dat","_")+"time_hops.png", bbox_inches='tight')
        plt.close()

    def _ene_angle_hops(self, hop_10, hop_01, mecp_10, mecp_01, ene_name, alpha_name):
        plt.rcParams['font.size'] = self.fs_rcParams
        for i in hop_10:
            hop_10 = plt.scatter(i[0],i[1],color='green', marker='o',s=35)
        for i in hop_01:
            hop_01 = plt.scatter(i[0],i[1],color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([0, 180])
        plt.ylim([0, 3])
        plt.ylabel('$\mathbf{E_1 - E_0 (eV)}$', fontsize = self.fs_ylabel)
        plt.xlabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = self.fs_xlabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+alpha_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+alpha_name.replace(".dat","")+"_hops.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+ene_name.replace(".dat","_")+alpha_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
            plt.savefig("lz_"+ene_name.replace(".dat","_")+alpha_name.replace(".dat","")+"_hops.png", bbox_inches='tight')
        plt.close()

    def plot_ene_dihedral_hops(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        ene_name = "e_gap.dat"
        pyr_name = "pyr_3210.dat"
        dihe_name = "dihe_2014.dat"
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        ene = self.no_time(ene_name) 
        plt.rcParams['font.size'] = self.fs_rcParams
        mecp_10 = []
        mecp_01 = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                ene_d = ene[j,i]
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([phi,ene_d])
                    else:
                        hop_10 = plt.scatter(phi,ene_d,color='green', marker='o',s=35)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([phi,ene_d])
                    else:
                        hop_01 = plt.scatter(phi,ene_d,color='red', marker='o',s=35)
        for i in mecp_10:
            mecp_10 = plt.scatter(i[0],i[1],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            mecp_01 = plt.scatter(i[0],i[1],color='red', marker='*',s=350, edgecolors='black')
        plt.xlim([-180, 180])
        plt.ylim([0, 3])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        plt.ylabel('$\mathbf{E_1 - E_0 (eV)}$', fontsize = self.fs_ylabel)
        plt.xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize = self.fs_xlabel)
        handles =[hop_10,hop_01,mecp_10,mecp_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
            plt.savefig("fssh_"+ene_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+ene_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.pdf", bbox_inches='tight')
            plt.savefig("lz_"+ene_name.replace(".dat","_")+dihe_name.replace(".dat","")+"_hops.png", bbox_inches='tight')
        plt.close()

    def plot_histogram_hops(self, n_bins=8):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        ene_name = "e_gap.dat"
        pyr_name = "pyr_3210.dat"
        dihe_name = "dihe_2014.dat"
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        ene = self.no_time(ene_name) 
        hop_10 = []
        hop_01 = []
        mecp_10 = []
        mecp_01 = []
        hop_10_hcnh = []
        hop_01_hcnh = []
        hop_10_cnh = []
        hop_01_cnh = []
        hop_10_pyr = []
        hop_01_pyr = []
        ene_z = []
        dihe_x = []
        angle_y = []
        for j in range(1,mdsteps):   #time_steps 
            for i in range(trajs):          #trajectories
                x = time[j]
                ene_d = ene[j,i]
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                ene_z.append(ene_d)
                dihe_x.append(phi)
                angle_y.append(alpha)
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append(x)
                    else:
                        hop_10.append(x)
                        hop_10_hcnh.append(phi)
                        hop_10_cnh.append(alpha)
                        hop_10_pyr.append(pyram)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append(x)
                    else:
                        hop_01.append(x)
                        hop_01_hcnh.append(phi)
                        hop_01_cnh.append(alpha)
                        hop_01_pyr.append(pyram)
        #self._1d_histogram(hop_10,hop_01,n_bins)
        #self._2d_histogram(hop_10,hop_01,hop_10_hcnh,hop_01_hcnh,n_bins)
        self._2d_surf_ene_dihedral_angle(dihe_x, angle_y, ene_z)

    def _2d_histogram(self,hop_10_x,hop_01_x,hop_10_y,hop_01_y,n_bins, x_type="time", y_type="hcnh"):
        plt.rcParams['font.size'] = '14'
        #plt.rcParams['axes.labelpad'] = 9
        fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
        ax = fig.add_subplot(111, projection='3d')
        plt.xlim([self.t_0, self.t_max])
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        bins_1 = [x for x in range(-180, 180+1,int((180+180)/n_bins))]
        hist, xedges, yedges = np.histogram2d(hop_10_x,hop_10_y, bins=[bins,bins_1])
        hist_1, xedges_1, yedges_1 = np.histogram2d(hop_01_x,hop_01_y, bins=[bins,bins_1])
        #hist, xedges, yedges = np.histogram2d([hop_10_x,hop_01_x],[hop_10_y,hop_01_y], bins=bins)

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = (xedges [1] - xedges [0])/3
        dy = (yedges [1] - yedges [0])/3
        dz = hist.flatten()

        xpos_1, ypos_1 = np.meshgrid(xedges_1[:-1] + 0.25 + dx, yedges_1[:-1] - 0.25 - dy, indexing="ij")
        xpos_1 = xpos_1.ravel()
        ypos_1 = ypos_1.ravel()
        zpos_1 = 0

        dx_1 = (xedges_1 [1] - xedges_1 [0])/3
        dy_1 = (yedges_1 [1] - yedges_1 [0])/3
        dz_1 = hist_1.flatten()

        #print("10: ",hist, xedges, yedges)
        #print("10: ",xpos, ypos, zpos, dx, dy, dz)
        #print("01: ",hist_1, xedges_1, yedges_1)
        #print("01: ",xpos_1, ypos_1, zpos_1, dx_1, dy_1, dz_1)
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color="green", zsort='max', alpha=0.4, edgecolor='black')
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green")
        ax.bar3d(xpos_1, ypos_1, zpos_1, dx_1, dy_1, dz_1, color="red", zsort='max', alpha=0.5, edgecolor='black')
        red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red")
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        ax.legend([green_proxy,red_proxy],labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), prop={'size': 12}, ncol=2)
        ax.set_xlabel('Time (fs)', fontsize=12,fontweight = 'bold',rotation=150)
        ax.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=12,fontweight = 'bold')
        # change fontsize
        ax.zaxis.set_tick_params(labelsize=12, pad=0)
        ax.xaxis.set_tick_params(labelsize=12, pad=0)
        ax.yaxis.set_tick_params(labelsize=12, pad=0)
        # disable auto rotation
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('Number of Hops', fontsize=12,fontweight = 'bold', rotation=90, labelpad=0)
        ax.view_init(30, 230)
        if self.label =="fssh":
            plt.savefig("fssh_"+x_type+"_"+y_type+"_hops.pdf", bbox_inches='tight', pad_inches = 0.4)
            plt.savefig("fssh_"+x_type+"_"+y_type+"_hops.png", bbox_inches='tight', pad_inches = 0.4)
        elif self.label =="lz":
            plt.savefig("lz_"+x_type+"_"+y_type+"_hops.pdf", bbox_inches='tight', pad_inches = 0.4)
            plt.savefig("lz_"+x_type+"_"+y_type+"_hops.png", bbox_inches='tight', pad_inches = 0.4)
        plt.close()

    def _1d_histogram(self,hop_10,hop_01,n_bins):
        plt.rcParams['font.size'] = self.fs_rcParams
        plt.xlim([self.t_0, self.t_max])
        #plt.ylim([-180, 180])
        #handles =[hop_10,hop_01,mecp_10,mecp_01]
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        n_plt, bins_plt, patches = plt.hist([hop_10,hop_01], bins = bins, color = ["green","red"], label=labels)
        plt.ylabel('Number of Hops', fontweight = 'bold', fontsize = self.fs_ylabel)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = self.fs_xlabel)
        if self.legend == "yes":
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_number_of_hops_time.pdf", bbox_inches='tight')
            plt.savefig("fssh_number_of_hops_time.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_number_of_hops_time.pdf", bbox_inches='tight')
            plt.savefig("lz_number_of_hops_time.png", bbox_inches='tight')
        plt.close()
        print("Histogram Information:", n_plt, bins_plt, patches)

    def _2d_surf_ene_dihedral_angle(self, x, y, z, x_type="dihedral", y_type="angle"):
        df = DataFrame({'x': x, 'y': y, 'z': z})
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set(xlim=(-180, 180), ylim=(0, 180), zlim=(0, 8), xlabel='$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', 
        ylabel='$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', zlabel=' ')

        #ax.view_init(30, 230)
        #ax.view_init(33, 45)
        ax.view_init(90, 90)
        plt.savefig("fssh_energy_"+x_type+"_"+y_type+".pdf", bbox_inches='tight', pad_inches = 0.4)
        plt.savefig("fssh_energy_"+x_type+"_"+y_type+".png", bbox_inches='tight', pad_inches = 0.4)
        plt.close()

    def plot_dihedral_angle_map_hops(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        ene_name = "e_gap.dat"
        pyr_name = "pyr_3210.dat"
        dihe_name = "dihe_2014.dat"
        angle_name = "angle_014.dat"
        pop_name = "pop.dat"
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        ci_alpha, ci_phi = self.con_int_two_angles("angle_014.dat", "dihe_2014.dat")
        angle_p = self.no_time("dihe_3014.dat") 
        pyr = self.no_time(pyr_name) 
        ene = self.no_time(ene_name) 
        plt.rcParams['font.size'] = 15
        mecp_10 = []
        mecp_01 = []
        hop_10 = []
        hop_01 = []
        ene_z = []
        dihe_x = []
        angle_y = []
        for i in range(trajs):          #trajectories
            for j in range(1,mdsteps):   #time_steps 
                ene_d = ene[j,i]
                pyram = pyr[j,i]
                y_dihe_1 = angle_p[j,i] 
                alpha = ci_alpha[j,i]
                phi = ci_phi[j,i]
                ene_z.append(ene_d)
                dihe_x.append(phi)
                angle_y.append(alpha)
                if hop[j-1,i]==1 and hop[j,i]==0:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_10.append([phi,alpha])
                    else:
                        hop_10.append([phi,alpha])
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    if ((self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(phi),self.ci[1],self.err)) or 
                        (self.compare(alpha,self.ci[0],self.err) and self.compare(np.abs(y_dihe_1),self.ci[1],self.err))): 
                        mecp_01.append([phi,alpha])
                    else:
                        hop_01.append([phi,alpha])

        f1 = plt.figure(figsize=(6,4))
        ax1 = f1.add_subplot(111)
        ax1.tripcolor(angle_y,dihe_x,ene_z)
        surf = ax1.tricontourf(angle_y,dihe_x,ene_z, cmap=cm.jet, antialiased=False)
        clb = f1.colorbar(surf, shrink=0.5, aspect=5, orientation = "vertical", format="%.0f", fraction=0.1)  # Add a colorbar to a plot
        clb.set_label('$\mathbf{E_1-E_0 (eV)}$', labelpad=0, y=1.2, rotation=0, fontsize = 15)
        clb.ax.tick_params(labelsize=15)

        for i in hop_10:
            h_10 = plt.scatter(i[1],i[0],color='green', marker='o',s=35, edgecolors='black')
        for i in hop_01:
            h_01 = plt.scatter(i[1],i[0],color='red', marker='o',s=35, edgecolors='black')
        for i in mecp_10:
            m_10 = plt.scatter(i[1],i[0],color='green', marker='*',s=350, edgecolors='black')
        for i in mecp_01:
            m_01 = plt.scatter(i[1],i[0],color='red', marker='*',s=350, edgecolors='black')

        plt.ylim([-180, 180])
        plt.xlim([0, 180])
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize = self.fs_ylabel)
        plt.xlabel('$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', fontsize = self.fs_xlabel)
        handles =[h_10,h_01,m_10,m_01]
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$",r"MECP $S_1/S_0$",r"MECP $S_0/S_1$"]
        if self.legend == "yes":
            plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.25), prop={'size': 12}, ncol=2)
        else:
            plt.legend().remove()
        if self.label =="fssh":
            plt.savefig("fssh_"+dihe_name.replace(".dat","_")+angle_name.replace(".dat","")+"_map_hops.pdf", bbox_inches='tight')
            plt.savefig("fssh_"+dihe_name.replace(".dat","_")+angle_name.replace(".dat","")+"_map_hops.png", bbox_inches='tight')
        elif self.label =="lz":
            plt.savefig("lz_"+dihe_name.replace(".dat","_")+angle_name.replace(".dat","")+"_map_hops.pdf", bbox_inches='tight')
            plt.savefig("lz_"+dihe_name.replace(".dat","_")+angle_name.replace(".dat","")+"_map_hops.png", bbox_inches='tight')
        plt.close()


    def plot_dihedral_dist_hops(self):
        prop = self.read_prop()
        nstates = prop.nstates
        var = self.read_db()   
        time_dihe, dihe_2357 = self.get_all_var(var.dihe_2357)
        time_12, dis_12 = self.get_all_var(var.dis_r12)
        time_23, dis_23 = self.get_all_var(var.dis_r23)
        time_35, dis_35 = self.get_all_var(var.dis_r35)
        time_57, dis_57 = self.get_all_var(var.dis_r57)
        time_70, dis_70 = self.get_all_var(var.dis_r70)
        time_01, dis_01 = self.get_all_var(var.dis_r01)
        time, hop = self.get_all_var(var.popu)
        row, col = dis_12.shape
        ep_10 = 0.05
        ep_12 = 0.08
        ep_12_pri = 0.05
        mecp_10 = []
        mecp_12 = []
        mecp_12_pri = []
        for i in range(col):         #trajectories
            for j in range(row-1):   #time_steps 
                x = np.array(time_12)[j]
                y_dihe_1 = np.array(dihe_2357)[j,i]
                y_12 = np.array(dis_12)[j,i]
                y_23 = np.array(dis_23)[j,i]
                y_35 = np.array(dis_35)[j,i]
                y_57 = np.array(dis_57)[j,i]
                y_70 = np.array(dis_70)[j,i]
                y_01 = np.array(dis_01)[j,i]
                if hop[j,i]==1 and hop[j+1,i]==0:
                    if (self.compare(y_12, 2.14, 0.04) and self.compare(y_23, 1.41, ep_10) and 
                        self.compare(y_35, 1.40, ep_10) and self.compare(y_57, 1.39, ep_10) and     
                            self.compare(y_70, 1.42, ep_10) and self.compare(y_01, 1.39, ep_10)):
                        print("S1-S0",x,y_12,y_23,y_35,y_57,y_70,y_01)
                        mecp_10.append([y_12,y_dihe_1])
                    else:
                        plt.scatter(y_12,y_dihe_1,color='green', marker='o',s=35)
                elif hop[j,i]==2 and hop[j+1,i]==1:
                    plt.scatter(y_12,y_dihe_1,color='gold', marker='o',s=35)
                elif hop[j,i]==0 and hop[j+1,i]==1:
                    plt.scatter(y_12,y_dihe_1,color='red', marker='o',s=35)
                elif hop[j,i]==1 and hop[j+1,i]==2:
                    if (self.compare(y_12, 2.03, 0.04) and self.compare(y_23, 1.42, ep_12) and 
                        self.compare(y_35, 1.45, ep_12) and self.compare(y_57, 1.37, ep_12) and 
                            self.compare(y_70, 1.45, ep_12) and self.compare(y_01, 1.42, ep_12)):
                        print("S1-S2",x,y_dihe_1,y_23,y_35,y_57,y_70,y_01)
                        mecp_12.append([y_12,y_dihe_1])
                    elif (self.compare(y_12, 2.36, 0.04) and self.compare(y_23, 1.41, ep_12_pri) and 
                        self.compare(y_35, 1.39, ep_12_pri) and self.compare(y_57, 1.42, ep_12_pri) and 
                            self.compare(y_70, 1.39, ep_12_pri) and self.compare(y_01, 1.41, ep_12_pri)):
                        print("S1-S2_pri",x,y_dihe_1,y_23,y_35,y_57,y_70,y_01)
                        mecp_12_pri.append([y_12,y_dihe_1])
                    else:
                        plt.scatter(y_12,y_dihe_1,color='blue', marker='o',s=35)
        for i in mecp_10:
            plt.scatter(i[0],i[1],color='green', marker='*',s=250, edgecolors='black') 
        for i in mecp_12:
            plt.scatter(i[0],i[1],color='blue', marker='*',s=250, edgecolors='black')
        for i in mecp_12_pri:
            plt.scatter(i[0],i[1],color='pink', marker='*',s=250, edgecolors='black')
        plt.xlim([1.3, 3])
        plt.ylim([-62, 82])
        plt.xlabel('$\mathbf{C_1-C_6 (\AA)}$', fontsize = 16) 
        plt.ylabel('$\mathbf{\sphericalangle C_1C_2C_3C_4 (degrees)}$', fontsize = 16) 
        plt.scatter(-5,-5,color='green', marker='o',s=35, label = r"$S_1$ $\rightarrow$ $S_0$")
        plt.scatter(-5,-5,color='gold', marker='o',s=35, label = r"$S_2$ $\rightarrow$ $S_1$")
        plt.scatter(-5,-5,color='red', marker='o',s=35, label = r"$S_0$ $\rightarrow$ $S_1$")
        plt.scatter(-5,-5,color='blue', marker='o',s=35, label = r"$S_1$ $\rightarrow$ $S_2$")
        plt.scatter(-5,-5,color='green', marker='*',s=250, edgecolors='black', label = r"MECP $S_1/S_0$")
        plt.scatter(-5,-5,color='blue', marker='*',s=250, edgecolors='black', label = r"MECP $S_1/S_2$")
        plt.scatter(-5,-5,color='pink', marker='*',s=250, edgecolors='black', label = r"MECP$^{\prime}$ $S_1/S_2$")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("Dihe_2357_dis12_hops.pdf", bbox_inches='tight')
        plt.close()

    def plot_r12_distances_time(self):
        prop = self.read_prop()
        nstates = prop.nstates
        var = self.read_db()   
        time_12, dis_12 = self.get_all_var(var.dis_r12)
        row, col = dis_12.shape
        for i in range(col):         #trajectories
            x = time_12
            y = np.array(dis_12)[:,i]
            plt.plot(x,y,c="red", label=' ', linewidth=0.5)
        plt.xlim([0, 350])
        plt.ylim([0, 6.5])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{C_1-C_6 (\AA)}$', fontsize = 16) 
        plt.savefig("R12_distances_time.pdf", bbox_inches='tight')
        plt.close()
    
    def high_180(self, y):
        y_nan = y.copy()
        for i in range(len(y)-1):
            if (y[i]-y[i+1])>0 and y[i]>179 and y[i+1]<0:
                y_nan[i] = y[i]*np.nan
            elif (y[i]-y[i+1])<0 and y[i]<-179 and y[i+1]>0:
                y_nan[i] = y[i]*np.nan
        return y_nan

    def plot_energies_diff_time(self):
        prop = self.read_prop()
        trajs = prop.trajs
        mdsteps = prop.mdsteps
        nstates = prop.nstates
        etot_name = "etot.dat"
        etot_diff = read_csv(etot_name)
        time = etot_diff['time']
        time = time.to_numpy()
        etot = etot_diff.to_numpy()[:,1:] # removing time column
        plt.rcParams['font.size'] = self.fs_rcParams
        plt.axhline(0,linestyle=':', c = 'red')
        ene = []
        t = []
        for i in range(trajs):          #trajectories
            for j in range(0,mdsteps+1):   #time_steps 
                ene.append(etot[j,i])
                t.append(time[j])
            plt.plot(t,ene,c="blue", label=' ', linewidth=0.5)
        plt.xlim([0, 200])
        plt.ylim([-1, 1])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{\Delta(E_{t_i}-E_{t_0})(eV)}$', fontsize = 16)
        plt.savefig("ene_tot_diff_time.pdf", bbox_inches='tight')
        plt.savefig("ene_tot_diff_time.png", bbox_inches='tight')
        plt.close()
                

    def plot_dihe2357_dihedrals_time(self):
        prop = self.read_prop()
        nstates = prop.nstates
        var = self.read_db()   
        time_dihe, dihe_2357 = self.get_all_var(var.dihe_2357)
        row, col = dihe_2357.shape
        for i in range(col):         #trajectories
            x = time_dihe
            y = np.array(dihe_2357)[:,i]
            y_nan = self.high_180(y)
            plt.plot(x,y_nan,c="red", label=' ', linewidth=0.5)
            #plt.plot(x,y_red,c="red", label=' ', linewidth=0.5)
        plt.xlim([0, 350])
        plt.ylim([-185, 185])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{\sphericalangle C_1C_2C_3C_4 (degrees)}$', fontsize = 16) 
        plt.savefig("Dihe_2357_dihedrals_time.pdf", bbox_inches='tight')
        plt.close()

    def plot_r12_hops_time(self):
        prop = self.read_prop()
        nstates = prop.nstates
        var = self.read_db()   
        time_12, dis_12 = self.get_all_var(var.dis_r12)
        time_23, dis_23 = self.get_all_var(var.dis_r23)
        time_35, dis_35 = self.get_all_var(var.dis_r35)
        time_57, dis_57 = self.get_all_var(var.dis_r57)
        time_70, dis_70 = self.get_all_var(var.dis_r70)
        time_01, dis_01 = self.get_all_var(var.dis_r01)
        time, hop = self.get_all_var(var.popu)
        row, col = dis_12.shape
        ep_10 = 0.05
        ep_12 = 0.08
        ep_12_pri = 0.05
        mecp_10 = []
        mecp_12 = []
        mecp_12_pri = []
        for i in range(col):         #trajectories
            for j in range(row-1):   #time_steps 
                x = np.array(time_12)[j]
                y = np.array(dis_12)[j,i]
                y_23 = np.array(dis_23)[j,i]
                y_35 = np.array(dis_35)[j,i]
                y_57 = np.array(dis_57)[j,i]
                y_70 = np.array(dis_70)[j,i]
                y_01 = np.array(dis_01)[j,i]
                if hop[j,i]==1 and hop[j+1,i]==0:
                    if (self.compare(y, 2.14, 0.04) and self.compare(y_23, 1.41, ep_10) and 
                        self.compare(y_35, 1.40, ep_10) and self.compare(y_57, 1.39, ep_10) and     
                            self.compare(y_70, 1.42, ep_10) and self.compare(y_01, 1.39, ep_10)):
                        print("S1-S0",x,y,y_23,y_35,y_57,y_70,y_01)
                        mecp_10.append([x,y])
                    else:
                        plt.scatter(x,y,color='green', marker='o',s=35)
                elif hop[j,i]==2 and hop[j+1,i]==1:
                    plt.scatter(x,y,color='gold', marker='o',s=35)
                #elif hop[j,i]==2 and hop[j+1,i]==0:
                #    plt.scatter(x,y,color='green', marker='+',s=35)
                elif hop[j,i]==0 and hop[j+1,i]==1:
                    plt.scatter(x,y,color='red', marker='o',s=35)
                elif hop[j,i]==1 and hop[j+1,i]==2:
                    if (self.compare(y, 2.03, 0.04) and self.compare(y_23, 1.42, ep_12) and 
                        self.compare(y_35, 1.45, ep_12) and self.compare(y_57, 1.37, ep_12) and 
                            self.compare(y_70, 1.45, ep_12) and self.compare(y_01, 1.42, ep_12)):
                        print("S1-S2",x,y,y_23,y_35,y_57,y_70,y_01)
                        mecp_12.append([x,y])
                    elif (self.compare(y, 2.36, 0.04) and self.compare(y_23, 1.41, ep_12_pri) and 
                        self.compare(y_35, 1.39, ep_12_pri) and self.compare(y_57, 1.42, ep_12_pri) and 
                            self.compare(y_70, 1.39, ep_12_pri) and self.compare(y_01, 1.41, ep_12_pri)):
                        print("S1-S2_pri",x,y,y_23,y_35,y_57,y_70,y_01)
                        mecp_12_pri.append([x,y])
                    else:
                        plt.scatter(x,y,color='blue', marker='o',s=35)
                #elif hop[j,i]==0 and hop[j+1,i]==2:
                #    plt.scatter(x,y,color='green', marker='o',s=35)
        for i in mecp_10:
            plt.scatter(i[0],i[1],color='green', marker='*',s=250, edgecolors='black') 
        for i in mecp_12:
            plt.scatter(i[0],i[1],color='blue', marker='*',s=250, edgecolors='black')
        for i in mecp_12_pri:
            plt.scatter(i[0],i[1],color='pink', marker='*',s=250, edgecolors='black')
        plt.xlim([0, 350])
        plt.ylim([1.3, 3])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{C_1-C_6 (\AA)}$', fontsize = 16) 
        plt.scatter(-5,-5,color='green', marker='o',s=35, label = r"$S_1$ $\rightarrow$ $S_0$")
        plt.scatter(-5,-5,color='gold', marker='o',s=35, label = r"$S_2$ $\rightarrow$ $S_1$")
        #plt.scatter(-5,-5,color='green', marker='+',s=35, label = r"$S_2$ $\rightarrow$ $S_0$")
        plt.scatter(-5,-5,color='red', marker='o',s=35, label = r"$S_0$ $\rightarrow$ $S_1$")
        plt.scatter(-5,-5,color='blue', marker='o',s=35, label = r"$S_1$ $\rightarrow$ $S_2$")
        #plt.scatter(-5,-5,color='green', marker='o',s=35, label = r"$S_0$ $\rightarrow$ $S_2$")
        plt.scatter(-5,-5,color='green', marker='*',s=250, edgecolors='black', label = r"MECP $S_1/S_0$")
        plt.scatter(-5,-5,color='blue', marker='*',s=250, edgecolors='black', label = r"MECP $S_1/S_2$")
        plt.scatter(-5,-5,color='pink', marker='*',s=250, edgecolors='black', label = r"MECP$^{\prime}$ $S_1/S_2$")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("R12_hops_time.pdf", bbox_inches='tight')
        plt.close()

    def write_csvs(self):
        var  = self.read_db()
        for value in var._fields:
        #for value in ["p_c0", "p_c1", "pop"]:
            print("Writhing csv files:", value)
            t, m = self.get_all_var(getattr(var, value))
            dct = {'time': t}
            for i in range(m.shape[1]):
                dct[i] = None
            filename = value + '.dat'
            with open(filename, 'w') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(dct.keys()))
                writer.writeheader()
                for i in tqdm(range(m.shape[0])):
                    dct = {'time': t[i]}
                    for j in range(m.shape[1]):
                        dct[j] = m[i,j]
                    writer.writerow(dct)
                    sleep(0.0001)


if __name__=="__main__":
    skip = sys.argv[1]
    popu = Population(skip)
    #popu.write_csvs()
    #popu.plot_population_pop_c()
    #popu.plot_population_adi()
    #popu.plot_dihedral_hops_time()
    #popu.plot_ene_angle_hops()
    #popu.plot_ene_dihedral_hops()
    #popu.plot_angle_hops_time()
    #popu.plot_pyra_hops_time()
    #popu.plot_pyra_angle_hops()
    #popu.plot_histogram_hops(8)
    #popu.plot_dihedral_angle_map_hops()
    #popu.plot_energies_diff_time()
    #popu.plot_population_adi_fitted()
    popu.get_qy_popu()
    #popu.plot_angle_dihedral_hops()
    #print(popu.get_all_var())

    #popu.write_csv('dis_r12.dat', 'dis_r12')
    #popu.write_csv('dis_r25.dat', 'dis_r25')
    #popu.write_csv('dis_r14.dat', 'dis_r14')
    #popu.write_csv('dis_r13.dat', 'dis_r13')
    #popu.write_csv('angle_014.dat', 'angle_014')
    #popu.write_csv('dihe_3014.dat', 'dihe_3014')
    #popu.write_csv('dihe_2014.dat', 'dihe_2014')
    #popu.write_csv('energies.dat', 'etot')
    #popu.plot_population_dia()
    #popu.plot_r12_hops_time()
    #popu.plot_dihedral_hops_time()
    #popu.plot_dihedral_dist_hops()
    #popu.plot_r12_distances_time()
    #popu.plot_dihe2357_dihedrals_time()
