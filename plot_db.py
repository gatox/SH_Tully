import sys
import matplotlib.pyplot as plt
import numpy as np
from pysurf.database import PySurfDB

class PlotResults:
    
    def __init__(self, output):
        self.output = PySurfDB.load_database(output, read_only=True)
        self.natoms = self.output.dimensions["natoms"]
        self.nstates = self.output.dimensions["nstates"]
        self.fs = 0.02418884254
        self.ev = 27.211324570273
        self.aa = 0.529177208
        self.ene_min = np.array(self.output["energy"]).min()
        self.time = self.output["time"][0:]*self.fs
        self.dim = len(self.time)
        self.atom_1 = int(atom_1)
        self.atom_2 = int(atom_2)

    def read_prop(self):
        prop = open("prop.inp", 'r+')    
        for line in prop:
            if "instate" in line:
                self.instate = int(line.split()[2])
            elif "states" in line:
                self.states = []
                for i in range(self.nstates):
                   self.states.append(int(line.split()[2+i]))

    def plot_energy(self):
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            plt.plot(self.time,(ene-self.ene_min)*self.ev, label = '$S_%i$'%i)
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                plt.axvline(x=self.time[i],label=f"Hop:(S{int(curr[0])} to S{int(new[0])}) at {int(self.time[i][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Energy(eV)}$', fontsize = 16)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("energy.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

    def plot_population(self):
        for i in range(self.nstates):
            population = np.array(self.output["fosc"])[:,i] 
            plt.plot(self.time,population, label = '$S_%i$' %i)
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                plt.axvline(x=self.time[i],label=f"Hop:(S{int(curr[0])} to S{int(new[0])}) at {int(self.time[i][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("population.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

    def dis_dimer(self, atom_1, atom_2):
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        crd = self.output["crd"]
        dimer = []
        for i,m in enumerate(crd):
            dimer.append((np.sqrt(np.sum((np.array(m[atom_1])-np.array(m[atom_2]))**2)))*self.aa)
        return dimer

    def plot_dist_vs_time(self, atom_1, atom_2):
        dimer = self.dis_dimer(atom_1, atom_2)
        plt.plot(self.time,dimer, label = '$r_1-r_2$')
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                plt.axvline(x=self.time[i],label=f"Hop:(S{int(curr[0])} to S{int(new[0])}) at {int(self.time[i][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{d(r_1-r_2)\AA}$', fontsize = 16) 
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("dist_vs_time.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

    def plot_ene_vs_dis(self, atom_1, atom_2):
        dimer = self.dis_dimer(atom_1, atom_2)
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            plt.plot(dimer,(ene-self.ene_min)*self.ev, label = '$S_%i$' %i)
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                ene_old = np.array(self.output["energy"])[i,int(curr)]
                ene_new = np.array(self.output["energy"])[i,int(new)]
                ene_hop_old = (ene_old-self.ene_min)*self.ev
                ene_hop_new = (ene_new-self.ene_min)*self.ev
                plt.axvline(x=dimer[i], label=f"Hop:S{int(curr[0])}({ene_hop_old:>0.2f} eV) to S{int(new[0])}({ene_hop_new:>0.2f} eV); ({dimer[i]:>0.2f} $\AA$/{int(self.time[i][0])} fs)",\
                            linestyle='--', c = 'purple')
        plt.xlabel('$\mathbf{d(r_1-r_2)\AA}$', fontsize = 16) 
        plt.ylabel('$\mathbf{Energy(eV)}$', fontsize = 16)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("ener_vs_dist.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

if __name__=="__main__":
    output = sys.argv[1]    
    atom_1 = sys.argv[2]
    atom_2 = sys.argv[3]
    picture = PlotResults(output)
    picture.plot_energy()
    picture.plot_population()
    picture.plot_dist_vs_time(atom_1,atom_2)
    picture.plot_ene_vs_dis(atom_1,atom_2)
