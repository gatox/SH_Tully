import numpy as np
import random
import time

from collections import namedtuple
from abc import abstractmethod
from pysurf.spp import ModelBase, SurfacePointProvider
from pysurf.database import PySurfDB
from pysurf.colt import Colt

class VelocityVerletPropagator:

    def __init__(self, state):
        self.state = state
        self.t = self.state.t
        self.dt = self.state.dt
        self.t_max = self.dt*self.state.mdsteps 
        #self.t_max = 2.0*np.abs(self.state.crd/self.state.vel)
        if self.state.method == "Surface_Hopping":
            self.electronic = SurfaceHopping(self.state)
            if self.state.ncoeff[self.state.instate] == 0:
                raise SystemExit("Wrong population for initial state")
        elif self.state.method == "Born_Oppenheimer":  
            self.electronic = BornOppenheimer(self.state)
        self.results = PrintResults()

    def run(self):
        
        if self.t > self.t_max:
            raise SystemExit("Noting to be done")
        
        state = self.state       
        results = self.results 
        grad_old = self.electronic.setup(state)
        acce_old = self.accelerations(state, grad_old)

        results.print_head(state)
        while(self.t <= self.t_max):
            """updating coordinates"""
            crd_new = self.positions(state, acce_old, self.dt)
            """updating accelerations"""    
            grad_new = self.electronic.new_surface(state, results, crd_new, self.t, self.dt)
            acce_new = self.accelerations(state, grad_new)
            """updating velocities"""
            vel_new = self.velocities(state, acce_old, acce_new, self.dt)
            """updating variables"""
            acce_old = self.update_state(state, acce_new, crd_new, vel_new) 
            self.t += self.dt 
        results.print_bottom(state)


    def accelerations(self, state, grad):
        if np.isscalar(state.mass) and np.isscalar(grad[state.instate]):
            return -grad[state.instate]/state.mass
        else:
            gradient = grad[state.instate]
            acce = np.zeros(gradient.shape)
            for i,m in enumerate(state.mass):
                acce[i] = -gradient[i]/m
            return acce

    def positions(self, state, a_0, dt):
        return state.crd + state.vel*dt + 0.5*a_0*dt**2

    def velocities(self, state, a_0, a_1, dt):
        return state.vel + 0.5*(a_0 + a_1)*dt 

    def update_state(self, state, acce_new, crd_new, vel_new):
        state.crd = crd_new
        state.vel = vel_new
        acce_old = acce_new
        return acce_old

class BornOppenheimer:

    needed_properties = ["energy","gradient"]

    def __init__(self, state):
        self.nstates = state.nstates
        self.natoms = state.natoms
        self.spp = SurfacePointProvider.from_questions(["energy","gradient"], self.nstates, self.natoms, config ="spp.inp", atomids = state.atomids)

    @abstractmethod
    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

    def cal_ekin(self, mass, vel):
        ekin = 0
        if np.isscalar(mass) and np.isscalar(vel):
            ekin = 0.5*mass*vel**2
        else:
            for i, m in enumerate(mass):
                ekin += 0.5*m*np.dot(vel[i],vel[i])
        return ekin
    
    def setup(self, state):
        state.ene = self.get_energy(state.crd)
        grad = self.get_gradient(state.crd)
        state.epot = state.ene[state.instate]
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad

    def new_surface(self, state, results, crd_new, t, dt):
        grad_new = self.get_gradient(crd_new)
        results.print_bh_var(t, dt, state) #printing variables 
        results.save_db(t,state) #save variables in database
        state.ene = self.get_energy(crd_new)
        state.epot = state.ene[state.instate]
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad_new

class SurfaceHopping(BornOppenheimer):

    #needed_properties = ["energy", "gradient", "nacs"]

    def __init__(self, state):
        self.nstates = state.nstates
        self.mass = state.mass
        self.natoms = state.natoms
        self.prob_name = state.prob
        self.coupling = state.coupling
        if self.coupling == "nacs":
            needed_properties = ["energy", "gradient", "nacs"]
            self.spp = SurfacePointProvider.from_questions(["energy","gradient","nacs"], self.nstates, self.natoms, config ="spp.inp", atomids = state.atomids)
        elif self.coupling == "wf_overlap":
            needed_properties = ["energy", "gradient", "wf_overlap"]
            self.spp = SurfacePointProvider.from_questions(["energy","gradient","wf_overlap"], self.nstates, self.natoms, config ="spp.inp", atomids = state.atomids)

    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

    def get_coupling(self, crd):
        if self.coupling == "nacs":
            result = self.spp.request(crd, ['nacs'])
            return result['nacs']
        elif self.coupling == "wf_overlap":
            result = self.spp.request(crd, ['wf_overlap'])
            return result['wf_overlap']
            
    def get_ene_cou_grad(self, crd):
        h_mch = self.get_energy(crd)
        grad = self.get_gradient(crd) 
        ene, u = np.linalg.eigh(np.diag(h_mch))
        if self.coupling == "nacs":
            nac = self.get_coupling(crd)
            ene_cou_grad = namedtuple("ene_cou_grad", "ene u nac grad")
            return ene_cou_grad(ene, u, nac, grad)
        elif self.coupling == "wf_overlap":
            wf_ov = self.get_coupling(crd)
            ene_cou_grad = namedtuple("ene_cou_grad", "ene u wf_ov grad")
            return ene_cou_grad(ene, u, wf_ov, grad)

    def elec_density(self, state):
        c_mch = state.ncoeff
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch,dtype=np.complex128) 
        return np.outer(c_mch, c_mch.conj())

    def grad_diag(self, ene_cou_grad):
        g_mch = ene_cou_grad.grad
        u = ene_cou_grad.u
        g_diag = {}
        for i in range(self.nstates):
            g_diag.update({i:np.dot(u.T.conj()[i,:],u[:,i]).real*g_mch[i]})
        return g_diag

    def vk_coupl_matrix(self, state):
        vel = state.vel
        nac = state.nac
        vk = np.zeros((self.nstates,self.nstates))
        if np.isscalar(vel):
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if i < j:
                        vk[i,j] = vel*nac[i,j]
                        vk[j,i] = -vk[i,j]
        else:
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if i < j:
                        vk[i,j] = np.dot(vel.flatten(),nac[i,j].flatten())
                        vk[j,i] = -vk[i,j]
        return vk

    def cal_ekin(self, mass, vel):
        ekin = 0
        if np.isscalar(mass) and np.isscalar(vel):
            ekin = 0.5*mass*vel**2
        else:
            for i, m in enumerate(mass):
                ekin += 0.5*m*np.dot(vel[i],vel[i])
        return ekin

    def setup(self, state):
        ene_cou_grad = self.get_ene_cou_grad(state.crd)
        grad_old_diag = self.grad_diag(ene_cou_grad)
        state.ene = ene_cou_grad.ene
        state.epot = state.ene[state.instate]
        state.u = ene_cou_grad.u
        if self.coupling == "nacs":
            state.nac = ene_cou_grad.nac
            state.vk = self.vk_coupl_matrix(state)
        elif self.coupling == "wf_overlap":
            state.vk = ene_cou_grad.wf_ov 
        state.ekin = self.cal_ekin(state.mass, state.vel)
        state.rho = self.elec_density(state)
        return grad_old_diag

    def mch_propagator(self, h_mch, vk, dt):
        h_total = np.diag(h_mch) - 1j*(vk) 
        ene, u = np.linalg.eigh(h_total)
        p_mch = np.linalg.multi_dot([u, np.diag(np.exp( -1j * ene * dt)), u.T.conj()])
        return p_mch

    def elec_density_new(self, rho, p_mch):
        """ Computing propagation and new density:
            D and U are diagonal and unitary matrices of hamiltonian 
            rho_ij(t) = c_i(t)*c_j(t)^{*}
            as c(t+dt) = U*exp(-j*D*dt)*U.T*c(t),
            then rho_ij(t+dt) = c_i(t+dt)*c_j(t+dt)^{*}
                              = U*exp(-j*D*dt)*U.T*c_i(t)*c_j(t)^{*}*U*exp(j*D*dt)*U.T
                              = U*exp(-j*D*dt)*U.T*rho_ij(t)*U*exp(j*D*dt)*U.T
        """
        return np.linalg.multi_dot([p_mch, rho, p_mch.T.conj()])

    def diag_propagator(self, ene_cou_grad, dt, state):
        c_mch = state.ncoeff
        u_new = ene_cou_grad.u
        u = state.u
        ene = state.ene
        vk = state.vk
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch)
        c_diag = np.dot(u.T.conj(),c_mch)
        p_mch_new = self.mch_propagator(ene, vk, dt)
        p_diag_new = np.dot(u_new.T.conj() ,np.dot(p_mch_new,u))
        c_diag_new = np.dot(p_diag_new,c_diag) 
        diag_prop = namedtuple("diag_prop","c_diag, c_diag_new, p_diag_new")
        return diag_prop(c_diag, c_diag_new, p_diag_new)

    def hopping_probability(self, c_j_dt, c_i_dt, c_i_t, p_diag_ji, p_diag_ii): 
        prob_factor_1 = 1 - np.abs(np.dot(c_i_dt, c_i_dt.conj()))/np.abs(np.dot(c_i_t, c_i_t.conj())) 
        prob_factor_2_N = ( np.dot(c_j_dt, np.dot(p_diag_ji.conj(), c_i_t.conj()) ) ).real
        prob_factor_2_D = ( np.abs(np.dot(c_i_t, c_i_t.conj())) -\
                       ( np.dot(c_i_dt, np.dot(p_diag_ii.conj(), c_i_t.conj()) ) ).real)
        if prob_factor_2_D == 0:
            prob_ji = 0.0
        elif prob_factor_1*((prob_factor_2_N/prob_factor_2_D)) < 0:
            prob_ji = 0.0
        else:
            prob_ji = prob_factor_1*((prob_factor_2_N/prob_factor_2_D))
        return prob_ji
   
    def probabilities_tully(self, state, dt):
        rho_old = state.rho
        instate = state.instate
        ene = state.ene
        vk = state.vk
        h_total = np.diag(ene) - 1j*(vk)
        p_mch = self.mch_propagator(ene, vk, dt)
        probs = (2.0 * np.imag(rho_old[instate,:] * h_total[:,instate]) * dt)/(np.real(rho_old[instate,instate])) 
        probs[instate] = 0.0
        probs = np.maximum(probs, 0.0)
        tully = namedtuple("tully", "probs rho_old p_mch")
        return tully(probs, rho_old, p_mch) 
 
    def probabilities_diagonal(self, state, diag_prop):
        c_diag_dt = diag_prop.c_diag_new
        c_diag = diag_prop.c_diag
        p_diag_dt = diag_prop.p_diag_new
        probs = np.zeros(self.nstates)
        instate = state.instate
        for i in range(self.nstates):
            probs[i] = self.hopping_probability(c_diag_dt[i], c_diag_dt[instate],\
                                                c_diag[instate],p_diag_dt[i,instate], p_diag_dt[instate,instate])
        probs[instate] = 0.0
        return probs 

    def nac_average(self, state_old, state_new, nac_old, nac_new):
        #return nac_new[state_new,state_old]
        #return nac_old[state_new,state_old] 
        return (0.5)*(nac_old[state_new,state_old] + nac_new[state_new,state_old])
         
    def diff_ji(self, state_old, state_new, ene_new):
        return ene_new[state_old] - ene_new[state_new]
    
    def beta_ji(self, vel, nac_av):
        if np.isscalar(vel):
            return vel*nac_av
        else:
            return np.dot(vel.flatten(),nac_av.flatten())
    
    def alpha_ji(self, nac_av):
        if np.isscalar(self.mass):
            return 0.5*(nac_av**2)/self.mass
        else:
            alpha = 0.0
            for i, m in enumerate(self.mass):
                alpha += np.dot(nac_av[i], nac_av[i])/m
            return 0.5*alpha

    def new_velocity(self, state, gama_ji, nac_av):
        if np.isscalar(state.vel) and np.isscalar(self.mass):
            state.vel = state.vel - gama_ji*(nac_av/self.mass)
        else:
            for i, m in enumerate(self.mass):
                state.vel[i] = state.vel[i] - gama_ji*(nac_av[i]/m)

    def rescale_velocity(self, state, beta, alpha, diff, state_new, nac_av):
        if (beta**2 + 4*alpha*diff) < 0.0:
            """
            If this condition is satisfied, there is not hopping and 
            then the nuclear velocity simply are reversed.
            """
            gama_ji = beta/alpha
            self.new_velocity(state, gama_ji, nac_av)
        else:
            """
            If this condition is satisfied, a hopping from 
            current state to the first true state takes place 
            and the current nuclear velocity is ajusted in order 
            to preserve the total energy.
            """
            state.instate = state_new
            if beta < 0.0:
                gama_ji = (beta + np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                self.new_velocity(state, gama_ji, nac_av)
            else:
                gama_ji = (beta - np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                self.new_velocity(state, gama_ji, nac_av)

    def surface_hopping(self, state, ene_cou_grad, probs):            
        nac_new = ene_cou_grad.nac
        ene_new = ene_cou_grad.ene
        aleatory = random.uniform(0,1)
        acc_probs = np.cumsum(probs)
        hopps = np.less(aleatory, acc_probs)
        state_old = state.instate
        nac_old = state.nac
        if any(hopps):
            for i in range(self.nstates):
                if hopps[i]:
                    state_new = state.states[i]
                    break
            nac_av = self.nac_average(state_old,state_new,nac_old,nac_new)
            diff = self.diff_ji(state_old,state_new,ene_new)
            beta = self.beta_ji(state.vel, nac_av)
            alpha = self.alpha_ji(nac_av)
            self.rescale_velocity(state, beta, alpha, diff, state_new, nac_av) 
        sur_hop = namedtuple("sur_hop", "aleatory acc_probs")
        return sur_hop(aleatory, acc_probs[state.instate])

    def new_prob_grad(self, state, ene_cou_grad, dt):
        if self.prob_name == "diagonal":
            grad_new = self.grad_diag(ene_cou_grad)
            diag_prop = self.diag_propagator(ene_cou_grad, dt, state)
            probs = self.probabilities_diagonal(state, diag_prop)
            result = namedtuple("result","probs grad_new diag_prop") 
            return result(probs, grad_new, diag_prop)
        elif self.prob_name == "tully":
            tully= self.probabilities_tully(state, dt)
            probs = tully.probs
            grad_new = ene_cou_grad.grad 
            result = namedtuple("result","probs grad_new, tully") 
            return result(probs, grad_new, tully)
        else:
            raise SystemExit("A right probability method is not defined")

    def new_ncoeff(self, state, grad_probs):
        if self.prob_name == "diagonal":   
            state.ncoeff = np.dot(state.u, grad_probs.diag_prop.c_diag_new)
            state.rho = self.elec_density(state)
        elif self.prob_name == "tully":
            state.rho = self.elec_density_new(grad_probs.tully.rho_old, grad_probs.tully.p_mch)
            state.ncoeff = np.diag(grad_probs.tully.rho_old.real) 
        else:
            raise SystemExit("A right probability method is not defined") 

    def new_surface(self, state, results, crd_new, t, dt):
        ene_cou_grad = self.get_ene_cou_grad(crd_new)
        grad_probs = self.new_prob_grad(state, ene_cou_grad, dt)
        sur_hop = self.surface_hopping(state, ene_cou_grad, grad_probs.probs)
        results.print_var(t, dt, sur_hop, state) #printing variables 
        results.save_db(t,state) #save variables in database
        state.ene = ene_cou_grad.ene
        state.epot = state.ene[state.instate]
        state.u = ene_cou_grad.u
        self.new_ncoeff(state, grad_probs)
        if self.coupling == "nacs":
            state.nac = ene_cou_grad.nac
            state.vk = self.vk_coupl_matrix(state)
        elif self.coupling == "wf_overlap":
            state.vk = ene_cou_grad.wf_ov 
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad_probs.grad_new

class State(Colt):

    _questions = """ 
    # chosen parameters
    db_file = :: existing_file 
    t = 0.0 :: float
    dt = 1.0 :: float
    mdsteps = 40000 :: float
    # instate is the initial state: 0 = G.S, 1 = E_1, ...
    instate = 1 :: int
    nstates = 2 :: int
    states = 0 1 :: ilist
    ncoeff = 0.0 1.0 :: flist
    prob = tully :: str :: tully, diagonal     
    coupling = nacs :: str :: nacs, wf_overlap
    method = Surface_Hopping :: str :: Surface_Hopping, Born_Oppenheimer  
    """
    
    def __init__(self, crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, coupling, method, atomids):
        self.crd = crd
        self.natoms = len(crd)
        self.atomids = atomids
        self.vel = vel
        self.mass = mass
        self.t = t
        self.dt = dt
        self.mdsteps = mdsteps
        self.instate = instate
        self.nstates = nstates
        self.states = states
        self.ncoeff = ncoeff
        self.prob = prob
        self.coupling = coupling
        self.method = method
        self.ekin = 0
        self.epot = 0
        self.nac = {}
        self.ene = []
        self.vk = []
        self.u = []
        self.rho = []
        if np.isscalar(self.mass):
            self.natoms = 1
        elif isinstance(self.mass, np.ndarray) != True:
            self.natoms = np.array([self.mass])

    @classmethod
    def from_config(cls, config):
        crd, vel, mass, atomids = cls.read_db(config["db_file"])
        t = config['t']
        dt = config['dt']
        mdsteps = config['mdsteps']
        instate = config['instate']
        nstates = config['nstates']
        states = config['states']
        ncoeff = config['ncoeff']
        prob = config['prob']
        coupling = config['coupling']
        method = config['method']
        return cls(crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, coupling, method, atomids)  

    @staticmethod
    def read_db(db_file):
        db = PySurfDB.load_database(db_file, read_only=True)
        crd = np.copy(db['crd'][0])
        vel = np.copy(db['veloc'][0])
        atomids = np.copy(db['atomids'])
        mass = np.copy(db['masses'])
        return crd, vel, mass, atomids

    @classmethod
    def from_initial(cls, crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, coupling, method):
        return cls(crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, coupling, method)

class PrintResults:
 
    def __init__(self):
        self.large = 110
        self.large_bo = 108
        self.dash = '-' * self.large
        self.dash_bo = '-' * self.large_bo
        self.gen_results = open("gen_results.out", "w")
        self.hopping = []
        self.tra_time = time.time()

    def save_db(self, t, state):
        nstates = state.nstates
        if np.isscalar(state.crd):
            natoms = int(1)
        else:
            natoms = len(state.crd)
        if state.method == "Surface_Hopping":
            db = PySurfDB.generate_database("results.db", data=["crd","veloc","energy","time","ekin","epot","etot","fosc","currstate"], dimensions ={"natoms":natoms, "nstates":nstates})
            db.set("currstate",state.instate)
            db.set("fosc",state.ncoeff)
        if state.method == "Born_Oppenheimer":
            db = PySurfDB.generate_database("results.db", data=["crd","veloc","energy","time","ekin","epot","etot"], dimensions ={"natoms":natoms, "nstates":nstates})
        db.set("crd",state.crd)
        db.set("veloc",state.vel)
        db.set("energy",state.ene)
        db.set("time",t)
        db.set("ekin",state.ekin)
        db.set("epot",state.epot)
        db.set("etot",state.ekin+state.epot)
        db.increase  # It increases the frame

    def dis_dimer(self, a,b):
        return np.sqrt(np.sum((a-b)**2))
    
    def print_acknowledgment(self, state):
        title = " Trajectory Surface Hopping Module "
        based = " This module uses the tools implemented in PySurf"
        contributors = " Module implemented by: Edison Salazar, Maximilian Menger, and Shirin Faraji "
        vel = state.vel
        crd = state.crd
        prob = state.prob
        coupling = state.coupling
        dt = state.dt
        mdsteps = state.mdsteps
        instate = state.instate
        self.instate = instate
        nstates = state.nstates
        ncoeff = state.ncoeff
        ack = namedtuple("ack", "title vel crd based actors prob coupling dt mdsteps instate nstates ncoeff")
        return ack(title, vel, crd, based, contributors, prob, coupling, dt, mdsteps, instate, nstates, ncoeff)

    def print_head(self, state):
        ack = self.print_acknowledgment(state)  
        if state.method == "Surface_Hopping":
            self.gen_results.write(f"\n{ack.title:=^{self.large}}\n")
            self.gen_results.write(f"\n{ack.based:^{self.large}}\n")
            self.gen_results.write(f"{ack.actors:^{self.large}}\n")        
            self.gen_results.write(f"\nInitial parameters:\n")
            self.gen_results.write(f"   Time step: {ack.dt}\n")
            self.gen_results.write(f"   MD steps: {ack.mdsteps}\n")
            self.gen_results.write(f"   Number of states: {ack.nstates}\n")
            self.gen_results.write(f"   Initial population: {ack.ncoeff}\n")
            self.gen_results.write(f"   Initial state: {ack.instate}\n")
            self.gen_results.write(f"   Probability method: {ack.prob}\n")
            self.gen_results.write(f"   Coupling: {ack.coupling}\n")
            self.gen_results.write(f"Computing a trajectory surface hopping simulation:\n")
            self.gen_results.write(self.dash + "\n")
            head = namedtuple("head","steps t ekin epot etotal hopp random state")
            head = head("MD_steps", "Time", "E_kinetic", "E_potential", "E_total", "Hopping_P", "Random", "State")
            self.gen_results.write(f"{head.steps:>10s} {head.t:>10s} {head.ekin:>15s} {head.epot:>17s} {head.etotal:>13s}"\
                    f"{head.hopp:>15s} {head.random:>11s} {head.state:>11s} \n")
            self.gen_results.write(self.dash + "\n")
        elif state.method == "Born_Oppenheimer":
            self.t_crd_vel_ene_popu = open("t_crd_vel_ene_popu.csv", "w")
            self.gen_results.write(f"\n{ack.title:=^{self.large_bo}}\n")
            self.gen_results.write(f"\n{ack.based:^{self.large_bo}}\n")
            self.gen_results.write(f"{ack.actors:^{self.large_bo}}\n")        
            self.gen_results.write(f"Initial parameters:\n")
            self.gen_results.write(f"   Initial position:\n")
            self.gen_results.write(f"   {self.dis_dimer(ack.crd[0],ack.crd[1]):>0.4f}\n")
            self.gen_results.write(f"   Initial velocity:\n")
            self.gen_results.write(f"   {self.dis_dimer(ack.vel[0],ack.vel[1]):>0.4f}\n")
            self.gen_results.write(f"   Time step: {ack.dt}\n")
            self.gen_results.write(f"   MD steps: {ack.mdsteps}\n")
            self.gen_results.write(f"   Active state: {ack.instate}\n")
            self.gen_results.write(f"Computing a Born Oppenheimer simulation:\n")
            self.gen_results.write(self.dash_bo + "\n")
            head = namedtuple("head","steps t dis dis_vel ekin epot etotal state")
            head = head("MD_steps", "Time", "D_r1-r2", "D_v1-v2", "E_kinetic", "E_potential", "E_total", "State")
            self.gen_results.write(f"{head.steps:>10s} {head.t:>10s} {head.dis:>12s} {head.dis_vel:>12s}"\
                    f"{head.ekin:>15s} {head.epot:>17s} {head.etotal:>13s} {head.state:>11s} \n")
            self.gen_results.write(self.dash_bo + "\n")
            self.t_crd_vel_ene_popu.write(f"{head.t},{head.dis},{head.dis_vel},{head.ekin},"\
                    f"{head.epot},{head.etotal},{head.state}\n")

    def print_var(self, t, dt, sur_hop, state):        
        var = namedtuple("var","steps t ekin epot etotal hopp random state")
        var = var(int(t/dt),t,state.ekin,state.epot,state.ekin + state.epot,\
                    sur_hop.acc_probs,sur_hop.aleatory,state.instate)
        self.gen_results.write(f"{var.steps:>8.0f} {var.t:>12.2f} {var.ekin:>15.3f} {var.epot:>17.4f}"\
                    f"{var.etotal:>13.4f} {var.hopp:>15.5f} {var.random:>11.5f} {var.state:>11.0f}\n")
        if var.state != self.instate:
            self.hopping.append(f"Hopping from state {self.instate} to state {state.instate}"\
                                f" in step: {var.steps}, at the time step: {var.t}")
            self.instate = var.state

    def print_bh_var(self, t, dt, state):        
        var = namedtuple("var","steps t dis dis_vel ekin epot etotal state")
        var = var(int(t/dt),t,self.dis_dimer(state.crd[0],state.crd[1]),self.dis_dimer(state.vel[0],\
                    state.vel[1]),state.ekin,state.epot,state.ekin + state.epot,state.instate)
        self.gen_results.write(f"{var.steps:>8.0f} {var.t:>12.2f} {var.dis:>12.4f} {var.dis_vel:>12.4f}"\
                    f"{var.ekin:>15.3f} {var.epot:>17.4f} {var.etotal:>13.4f} {var.state:>11.0f}\n")
        self.t_crd_vel_ene_popu.write(f"{var.t:>0.3f},{var.dis:>0.8f},{var.dis_vel:>0.8f},"\
                    f"{var.ekin:>0.8f},{var.epot:>0.8f},{var.etotal:>0.8f},{var.state:>0.0f}\n")

    def print_bottom(self, state):
        if state.method == "Surface_Hopping":
            self.gen_results.write(self.dash + "\n")
            self.gen_results.write(f"Some important variables are printed in results.db\n")
            if self.hopping:
                for i in range(len(self.hopping)):
                    self.gen_results.write(f"{self.hopping[i]}\n")
                #if i == 0:
                #    print(f"There is {i+1} hop")
                #else:
                #    print(f"There are {i+1} hops")
            else:
                self.gen_results.write(f"No hoppings achieved\n")
        elif state.method == "Born_Oppenheimer": 
            self.t_crd_vel_ene_popu.close()
            self.gen_results.write(self.dash_bo + "\n")
            self.gen_results.write(f"Some important variables are printed in t_crd_vel_ene_popu.csv and results.db\n")
        time_seg = time.time()-self.tra_time
        day = time_seg // (24*3600)
        time_seg = time_seg % (24*3600)
        hour = time_seg // 3600
        time_seg %= 3600
        minutes = time_seg // 60
        time_seg %= 60
        seconds = time_seg
        self.gen_results.write(f"Total job time: {day:>0.0f}:{hour:>0.0f}:{minutes:>0.0f}:{seconds:>0.0f}\n")
        
        self.gen_results.write(f"{time.ctime()}")
        self.gen_results.close()

if __name__=="__main__":
    elec_state = State.from_questions(config = "prop.inp")
    DY = VelocityVerletPropagator(elec_state)    
    try:
        result_2 = DY.run()
    except SystemExit as err:
        print("An error:", err) 
