import numpy as np
import random
from collections import namedtuple
from abc import abstractmethod
from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
from tully_model_3 import Tully_3
from tully_model_4 import Tully_4
from pysurf.spp import ModelBase, SurfacePointProvider
from colt import Colt

class VelocityVerletPropagator:

    def __init__(self, t, state):
        self.t = t
        self.state = state
        self.dt = np.abs(0.05/self.state.vel)
        self.t_max = 2.0*np.abs(self.state.crd/self.state.vel)
        self.electronic = SurfaceHopping(self.state)
        self.results = PrintResults()

    def run(self):
        
        if self.t > self.t_max:
            raise SystemExit("Noting to be done")
        
        state = self.state       
        results = self.results 
        grad_old = self.electronic.setup(state)
        acce_old = self.accelerations(state, grad_old)

        results.print_head()
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
        results.print_bottom()

    def accelerations(self, state, grad):
        return -grad[state.instate]/state.mass

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
        self.spp = SurfacePointProvider.from_questions(["energy","gradient"], self.nstates, self.natoms, config ="model_BornOppenheimer.ini")

    @abstractmethod
    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']
    
    def setup(self, state):
        pass

class SurfaceHopping(BornOppenheimer):

    needed_properties = ["energy", "gradient", "coupling"]

    def __init__(self, state):
        self.nstates = state.nstates
        self.mass = state.mass
        self.natoms = state.natoms
        self.prob_name = state.prob
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, self.natoms, config ="model_SurfaceHopping.ini")

    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

    def get_coupling(self, crd):
        result = self.spp.request(crd, ['coupling'])
        return result['coupling']

    def get_ene_nac_grad(self, crd):
        h_mch = self.get_energy(crd)
        grad = self.get_gradient(crd)
        nac = self.get_coupling(crd)
        ene, u = np.linalg.eigh(np.diag(h_mch))
        ene_nac_grad = namedtuple("ene_nac_grad", "ene u nac grad")
        return ene_nac_grad(ene, u, nac, grad)

    def elec_density(self, state):
        c_mch = state.ncoeff
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch,dtype=np.complex128) 
        return np.outer(c_mch, c_mch.conj())

    def grad_diag(self, ene_nac_grad):
        g_mch = ene_nac_grad.grad
        u = ene_nac_grad.u
        g_diag = {}
        for i in range(self.nstates):
            g_diag.update({i:np.dot(u.T.conj()[i,:],u[:,i]).real*g_mch[i]})
        return g_diag

    def vk_coupl_matrix(self, vel, nac):
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
        ene_nac_grad = self.get_ene_nac_grad(state.crd)
        grad_old_diag = self.grad_diag(ene_nac_grad)
        state.ene = ene_nac_grad.ene
        state.epot = state.ene[state.instate]
        state.u = ene_nac_grad.u
        state.nac = ene_nac_grad.nac
        state.vk = self.vk_coupl_matrix(state.vel, state.nac)
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

    def diag_propagator(self, ene_nac_grad, dt, state):
        c_mch = state.ncoeff
        u_new = ene_nac_grad.u
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

    def surface_hopping(self, state, ene_nac_grad, probs):            
        nac_new = ene_nac_grad.nac
        ene_new = ene_nac_grad.ene
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

    def new_prob_grad(self, state, ene_nac_grad, dt):
        if self.prob_name == "diagonal":
            grad_new = self.grad_diag(ene_nac_grad)
            diag_prop = self.diag_propagator(ene_nac_grad, dt, state)
            probs = self.probabilities_diagonal(state, diag_prop)
            result = namedtuple("result","probs grad_new diag_prop") 
            return result(probs, grad_new, diag_prop)
        elif self.prob_name == "tully":
            tully= self.probabilities_tully(state, dt)
            probs = tully.probs
            grad_new = ene_nac_grad.grad 
            result = namedtuple("result","probs grad_new, tully") 
            return result(probs, grad_new, tully)
        else:
            raise SystemExit("A right probability method is not defined")

    def new_ncoeff(self, state, grad_probs):
        if self.prob_name == "diagonal":   
            state.ncoeff = np.dot(state.u, grad_probs.diag_prop.c_diag_new)
        elif self.prob_name == "tully":
            state.rho = self.elec_density_new(grad_probs.tully.rho_old, grad_probs.tully.p_mch)
            state.ncoeff = np.diag(grad_probs.tully.rho_old.real) 
        else:
            raise SystemExit("A right probability method is not defined") 

    def new_surface(self, state, results, crd_new, t, dt):
        ene_nac_grad = self.get_ene_nac_grad(crd_new)
        grad_probs = self.new_prob_grad(state, ene_nac_grad, dt)
        sur_hop = self.surface_hopping(state, ene_nac_grad, grad_probs.probs)
        results.print_var(t, dt, sur_hop, state) #printing variables 
        state.ene = ene_nac_grad.ene
        state.epot = state.ene[state.instate]
        state.u = ene_nac_grad.u
        self.new_ncoeff(state, grad_probs)
        state.nac = ene_nac_grad.nac
        state.vk = self.vk_coupl_matrix(state.vel, state.nac)
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad_probs.grad_new

class State(Colt):

    _questions = """ 
    # chosen parameters
    crd = -4.0 :: float
    vel = 1.0 :: float
    mass = 1.0 :: float
    # instate is the initial state: 0 = G.S, 1 = E_1, ...
    instate = 1 :: int
    nstates = 2 :: int
    states = 0 1 :: ilist
    ncoeff = 0.0 1.0 :: flist
    prob = tully :: str 
    """
    
    def __init__(self, crd, vel, mass, instate, nstates, states, ncoeff, prob):
        self.crd = crd
        self.vel = vel
        self.mass = mass
        self.instate = instate
        self.nstates = nstates
        self.states = states
        self.ncoeff = ncoeff
        self.prob = prob
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
        crd = config['crd']
        vel = config['vel']
        mass = config['mass']
        instate = config['instate']
        nstates = config['nstates']
        states = config['states']
        ncoeff = config['ncoeff']
        prob = config['prob']
        return cls(crd, vel, mass, instate, nstates, states, ncoeff, prob) 

    @classmethod
    def from_initial(cls, crd, vel, mass, instate, nstates, states, ncoeff, prob):
        return cls(crd, vel, mass, instate, nstates, states, ncoeff, prob)

class PrintResults:
 
    def __init__(self):
        self.dash = '-' * 141
        self.all_variables = open("all_variables.out", "w")
        self.t_vs_crd = open("t_vs_crd.out", "w")

    def print_head(self):
        self.all_variables.write(self.dash + "\n")
        Header = ["MD_steps", "Time", "Position", "Velocity", "E_kinetic",\
                     "E_potential", "E_total", "Hopping_P", "Random", "State"]
        self.all_variables.write(f"{Header[0]:>10s} {Header[1]:>10s} {Header[2]:>14s} {Header[3]:>14s}"\
                    f"{Header[4]:>15s} {Header[5]:>17s} {Header[6]:>13s} {Header[7]:>15s} {Header[8]:>11s}  {Header[9]:>11s} \n")
        self.all_variables.write(self.dash + "\n")
        self.t_vs_crd.write(f"#{Header[1]:>10s} {Header[2]:>12s} \n")

    def print_var(self, t, dt, sur_hop, state):
        Var = (int(t/dt)+1,t,state.crd,state.vel,state.ekin,state.epot,state.ekin + state.epot,sur_hop.acc_probs,sur_hop.aleatory,state.instate)
        self.all_variables.write(f"{Var[0]:>8.0f} {Var[1]:>12.1f} {Var[2]:>14.4f}"\
                    f"{Var[3]:>14.4f} {Var[4]:>15.3f} {Var[5]:>17.4f} {Var[6]:>13.4f} {Var[7]:>15.5f} {Var[8]:>11.5f} {Var[9]:>11.0f} \n")
        self.t_vs_crd.write(f"{Var[1]:>11.1f} {Var[2]:>12.4f} \n")
    
    def print_bottom(self):
        self.all_variables.write(self.dash)
        self.all_variables.close()
        self.t_vs_crd.close()

if __name__=="__main__":
    elec_state = State.from_questions(config = "state_setting.ini")
    DY = VelocityVerletPropagator(0, elec_state)    
    try:
        result_2 = DY.run()
    except SystemExit as err:
        print("An error:", err) 
