import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from jinja2 import Template #build templates
import numpy as np
from numpy import copy 
from pysurf.database import PySurfDB
from pysurf.system import Molecule
from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)


tpl = Template("""{{natoms}}
{% for atomid, crd in mol %} 
{{mol.format(atomid, crd)}} {% endfor %}

""")

bagel_ini = Template("""
{ "bagel" : [

""")

bagel_geo = Template("""
{
  "title" : "molecule",
  "basis" : "cc-pvdz",
  "df_basis" : "cc-pvdz-jkfit",
  "angstrom" : "false",
  "geometry" : [{% for atomid, crd in mol %}{% if not loop.last %}
                {{mol.bagel_xyz(atomid, crd)}},{% else %}
                {{mol.bagel_xyz(atomid, crd)}} {% endif %}{% endfor %}
  ]
 }

""")

bagel_end = Template("""
]}
""")

class PlotResults:
    
    tpl = tpl
    bagel_ini = bagel_ini
    bagel_geo = bagel_geo
    bagel_end = bagel_end
    
    def __init__(self, output):
        self.output = PySurfDB.load_database(output, read_only=True)
        self.nstates = self.output.dimensions["nstates"]
        self.fs = 0.02418884254
        self.ev = 27.211324570273
        self.aa = 0.529177208
        self.ene_min = np.array(self.output["energy"]).min()
        self.time = self.output["time"][0:]*self.fs
        self.dim = len(self.time)
        self.fs_rcParams = '20'

    def read_prop(self):
        prop = open("prop.inp", 'r+')    
        for line in prop:
            if "instate" in line:
                self.instate = int(line.split()[2])
            elif "states" in line:
                self.states = []
                for i in range(self.nstates):
                   self.states.append(int(line.split()[2+i]))

    def dis_dimer(self, atom_1, atom_2):
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        crd = self.output["crd"]
        dimer = []
        for i,m in enumerate(crd):
            dimer.append((np.sqrt(np.sum((np.array(m[atom_1])-np.array(m[atom_2]))**2)))*self.aa)
        return dimer

    def dihedral(self, a, b, c, d):
        crd = self.output["crd"]
        torsion = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)])
            vec_b = np.array(m[int(b)])
            vec_c = np.array(m[int(c)])
            vec_d = np.array(m[int(d)])
            torsion.append(calc_dihedral(Vector(vec_a),Vector(vec_b),Vector(vec_c),Vector(vec_d))* 180 / np.pi)
        return torsion

    def angle(self, a, b, c):
        crd = self.output["crd"]
        angle = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)])
            vec_b = np.array(m[int(b)])
            vec_c = np.array(m[int(c)])
            angle.append(calc_angle(Vector(vec_a),Vector(vec_b),Vector(vec_c))* 180 / np.pi)
        return angle 

    def pyramidalization_angle(self, a, b, c, o):
        crd = self.output["crd"]
        angle = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)]) - np.array(m[int(o)])
            vec_b = np.array(m[int(b)]) - np.array(m[int(o)])
            vec_c = np.array(m[int(c)]) - np.array(m[int(o)])
            vec_u = np.cross(vec_a, vec_b)
            d_cu = np.dot(vec_c,vec_u)
            cr_cu = np.cross(vec_c, vec_u)
            n_cr_cu = np.linalg.norm(cr_cu)
            res = np.math.atan2(n_cr_cu,d_cu)
            angle.append(90 - np.degrees(res))
        return angle 

    def high_180(self, y):
        y_nan = y.copy()
        for i in range(1,int(len(y))):
            #if (y[i]-y[i+1])>0 and y[i]>179 and y[i+1]<0:
            #    y_nan[i] = y[i]*np.nan
            #elif (y[i]-y[i+1])<0 and y[i]<-179 and y[i+1]>0:
            #    y_nan[i] = y[i]*np.nan
            if y[i-1]>0 and y[i]<0: 
                y_nan[i] = y[i]*np.nan
            elif y[i-1]<0 and y[i]>0:
                y_nan[i] = y[i]*np.nan
        return y_nan

    def write_traj_movie_xyz(self):
        db_file = "sampling.db"
        db = PySurfDB.load_database(db_file, read_only=True)
        atomids = copy(db['atomids'])
        crd = self.output["crd"]
        natoms = len(atomids)
        molecule = Molecule(atomids, None) 
        filename = 'traj.xyz'
        traj_xyz = open(filename, 'w')
        for i in range(self.dim):
            molecule.crd = np.array(crd[i])*self.aa  # coordinates must be in Angstrom units 
            traj_xyz.write(self.tpl.render(natoms=natoms,mol=molecule))
        traj_xyz.close()

    def write_bagel_format(self):
        db_file = "sampling.db"
        db = PySurfDB.load_database(db_file, read_only=True)
        atomids = copy(db['atomids'])
        crd = self.output["crd"]
        natoms = len(atomids)
        molecule = Molecule(atomids, None) 
        filename = 'bagel.in'
        traj_xyz = open(filename, 'w')
        molecule.crd = np.array(crd[0])  # coordinates must be in Bohr units 
        traj_xyz.write(self.bagel_ini.render())
        traj_xyz.write(self.bagel_geo.render(natoms=natoms,mol=molecule))
        traj_xyz.write(self.bagel_end.render())
        traj_xyz.close()
        

    def plot_energies_diff(self):
        plt.rcParams['font.size'] = self.fs_rcParams
        etot = np.array(self.output["etot"])
        epot = np.array(self.output["epot"])
        ekin = np.array(self.output["ekin"])
        plt.axhline(0,linestyle=':', c = 'red')
        plt.plot(self.time,(etot-etot[0])*self.ev,color='blue',label = '$\Delta E_{tot}$') 
        plt.plot(self.time,(epot-epot[0])*self.ev,color='darkgreen',linestyle='-.', label = '$\Delta E_{pot}$') 
        plt.plot(self.time,(ekin-ekin[0])*self.ev,color='black',linestyle='--', label = '$\Delta E_{kin}$') 
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.time[i-1]
                plt.axvline(x[0],label=fr"Hop:($S_{int(curr[0])} \to S_{int(new[0])}$) at {int(x[0])} fs",linestyle='--', c = 'purple')
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{\Delta(E_{t_i}-E_{t_0})(eV)}$', fontsize = 16)
        plt.legend(loc='upper center', prop={'size': 14}, bbox_to_anchor=(0.5, 1.25), 
                    ncol=3, fancybox=True, shadow=False)
        plt.savefig("energies_diff.pdf", bbox_inches='tight')
        plt.savefig("energies_diff.png", bbox_inches='tight')
        plt.close()
        #return plt.show()

    def plot_energy(self):
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            plt.plot(self.time,(ene-self.ene_min)*self.ev, label = '$S_%i$'%i)
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.time[i-1]
                plt.axvline(x[0],label=f"Hop:(S_{int(curr[0])} to S_{int(new[0])}) at {int(x[0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Energy(eV)}$', fontsize = 16)
        plt.legend(loc='upper center', prop={'size': 14}, bbox_to_anchor=(0.5, 1.2), 
                    ncol=3, fancybox=True, shadow=False)
        plt.savefig("energy.pdf", bbox_inches='tight')
        plt.savefig("energy.png", bbox_inches='tight')
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
                plt.axvline(x=self.time[i+1],label=f"Hop:(S_{int(curr[0])} to S_{int(new[0])}) at {int(self.time[i+1][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend(loc='upper center', prop={'size': 12}, bbox_to_anchor=(0.5, 1.2), 
                    ncol=2, fancybox=True, shadow=False)
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

    def dihedral(self, a, b, c, d):
        crd = self.output["crd"]
        torsion = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)])
            vec_b = np.array(m[int(b)])
            vec_c = np.array(m[int(c)])
            vec_d = np.array(m[int(d)])
            torsion.append(calc_dihedral(Vector(vec_a),Vector(vec_b),Vector(vec_c),Vector(vec_d))* 180 / np.pi)
        return torsion

    def plot_dihedral_vs_time(self, atom_1, atom_2, atom_3, atom_4):
        dihedral = self.dihedral(atom_1, atom_2, atom_3, atom_4)
        dihedral_nan = self.high_180(dihedral)
        plt.plot(self.time,dihedral_nan, label = '$H_1-C_1-N_2-H_5$')
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                plt.axvline(x=self.time[i+1],label=f"Hop:(S_{int(curr[0])} to S_{int(new[0])}) at {int(self.time[i+1][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{\sphericalangle H_4C_1N_2H_5 (degrees)}$', fontsize = 16) 
        plt.legend(loc='upper center', prop={'size': 12}, bbox_to_anchor=(0.5, 1.2),
          ncol=1, fancybox=True, shadow=False)
        plt.savefig("dihedral_vs_time.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

    def plot_energy_angles_vs_time(self, atom_1, atom_2, atom_3, atom_4, atom_5):
        plt.rcParams['font.size'] = self.fs_rcParams
        plt.figure(figsize=(8,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # the first subplot
        ax0 = plt.subplot(gs[0])
        #ax0.text(0.05, 0.95, "i", transform=ax0.transAxes,
        #        fontsize=self.fs_rcParams, fontweight='bold', va='top')
        ax0.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 16)
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            line0, = ax0.plot(self.time,(ene-self.ene_min)*self.ev, label = '$S_%i$'%i)
        self.hop = []
        for i in range(1, self.dim):
            if i == 1:
                self.curr_t = self.output["currstate"][i-1]
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.time[i-1]
                line0 = ax0.axvline(x[0],label=fr"Hop:($S_{int(curr[0])} \to S_{int(new[0])}$) at {int(x[0])} fs",linestyle='--', c = 'purple')
                self.hop.append(int(i-1)) 
        if len(self.hop) == 1:  # only implemented for one hop
            t = self.hop[0]
            ene_1 = np.array(self.output["energy"])[:t,1] 
            ene_0 = np.array(self.output["energy"])[t:,0] 
            line0 = ax0.scatter(self.time[:t],(ene_1-self.ene_min)*self.ev, marker='o', facecolors='none', edgecolors='black')
            line0 = ax0.scatter(self.time[t:],(ene_0-self.ene_min)*self.ev, marker='o', facecolors='none', edgecolors='black')
            
        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex = ax0)
        #ax1.text(0.05, 0.95, "ii", transform=ax1.transAxes,
        #        fontsize=self.fs_rcParams, fontweight='bold', va='top')
        ax1.set_ylabel("Angle (degrees)", fontweight = 'bold', fontsize = 16)
        #ax1.set_ylim([-5, 115])
        dihedral = self.dihedral(atom_3, atom_1, atom_2, atom_5)
        dihedral_nan = self.high_180(dihedral)
        angle = self.angle(atom_1, atom_2, atom_5)
        pira = self.pyramidalization_angle(atom_3, atom_4, atom_2, atom_1)
        line1, = ax1.plot(self.time, dihedral_nan, color='blue', label = 'HCNH' )
        line1, = ax1.plot(self.time, angle, color='darkgreen',linestyle='--', label = 'CNH')
        line1, = ax1.plot(self.time, pira, color='black',linestyle='-.', label = 'pyram.')
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.time[i-1]
                line1 = ax1.axvline(x[0],linestyle='--', c = 'purple')
        plt.setp(ax0.get_xticklabels(), visible=False)

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), prop={'size': 14}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.savefig("Energy_angles_time.pdf", bbox_inches='tight')
        plt.savefig("Energy_angles_time.png", bbox_inches='tight')
        plt.close()


    def plot_dist_vs_time(self, atom_1, atom_2):
        dimer = self.dis_dimer(atom_1, atom_2)
        plt.plot(self.time,dimer, label = '$C_1-C_6$')
        for i in range(self.dim-1):
            if self.output["currstate"][i] != self.output["currstate"][i+1]:
                curr = self.output["currstate"][i]
                new = self.output["currstate"][i+1]
                plt.axvline(x=self.time[i+1],label=f"Hop:(S_{int(curr[0])} to S_{int(new[0])}) at {int(self.time[i+1][0])} fs",\
                            linestyle='--', c = 'purple')
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{C_1-C_6(\AA)}$', fontsize = 16) 
        plt.legend(loc='upper right', prop={'size': 12})
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
                plt.axvline(x=dimer[i+1], label=f"Hop:S_{int(curr[0])}({ene_hop_old:>0.2f} eV) to S_{int(new[0])}({ene_hop_new:>0.2f} eV); ({dimer[i+1]:>0.2f} $\AA$/{int(self.time[i+1][0])} fs)",\
                            linestyle='--', c = 'purple')
        plt.xlabel('$\mathbf{C_1-C_6(\AA)}$', fontsize = 16) 
        plt.ylabel('$\mathbf{Energy(eV)}$', fontsize = 16)
        plt.legend(loc='upper right', prop={'size': 12})
        plt.savefig("ener_vs_dist.pdf", bbox_inches='tight')
        plt.close()
        #return plt.show()

if __name__=="__main__":
    output = sys.argv[1]  #results.db  
    atom_1 = sys.argv[2]    
    atom_2 = sys.argv[3]
    atom_3 = sys.argv[4]
    atom_4 = sys.argv[5]
    atom_5 = sys.argv[6]
    picture = PlotResults(output)
    #picture.write_traj_movie_xyz()
    picture.write_bagel_format()
    #picture.plot_energies_diff()
    #picture.plot_energy()
    #picture.plot_population()
    #picture.plot_dihedral_vs_time(atom_1, atom_2, atom_3, atom_4)
    #picture.plot_energy_angles_vs_time(atom_1, atom_2, atom_3, atom_4, atom_5)
    #picture.plot_dist_vs_time(atom_1,atom_2)
    #picture.plot_ene_vs_dis(atom_1,atom_2)
