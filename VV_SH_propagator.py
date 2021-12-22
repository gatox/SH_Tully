import numpy as np
import random

from abc import abstractmethod
from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
from pysurf.spp import ModelBase, SurfacePointProvider
from colt import Colt

class VelocityVerletPropagator:

    def __init__(self, t, State):
        self.t = t
        self.state = State
        self.dt = np.abs(0.05/State.vel)
        self.t_max = 2.0*np.abs(State.crd/State.vel)
        self.electronic = SurfaceHopping(self.state)

    def run(self):
        
        if self.t > self.t_max:
            raise SystemExit("Noting to be done")
        
        state = self.state        
        grad_old = self.electronic.setup(state)
        acce_old = self.accelerations(state, grad_old)

        print_head()
        while(self.t <= self.t_max):
            """updating coordinates"""
            crd_new = self.positions(state, acce_old, self.dt)
            """updating accelerations"""    
            grad_new = self.electronic.new_surface(state, crd_new, self.t, self.dt)
            acce_new = self.accelerations(state, grad_new)
            """updating velocities"""
            vel_new = self.velocities(state, acce_old, acce_new, self.dt)
            """updating variables"""
            acce_old = self.update_state(state, acce_new, crd_new, vel_new) 
            self.t += self.dt 
        return print_bottom()

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

    needed_properties = ["energy","gradient","coupling"]

    def __init__(self, state):
        self.nstates = state.nstates
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

    @abstractmethod
    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def rescale_velocity(self, instate):
        pass

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

    def get_coupling(self, crd):
        result = self.spp.request(crd, ['coupling'])
        return result['coupling']
    
    def setup(self, state):
        pass

class SurfaceHopping(BornOppenheimer):

    needed_properties = ["energy", "gradient", "coupling"]

    def __init__(self, state):
        self.nstates = state.nstates
        self.mass = state.mass
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

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
        grad_old = self.get_gradient(crd)
        nac = self.get_coupling(crd)
        ene, u = np.linalg.eigh(np.diag(h_mch))
        return ene, u, nac, grad_old

    def grad_diag(self, g_mch, u):
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
        ene, u, nac, grad_old = self.get_ene_nac_grad(state.crd)
        grad_old_diag = self.grad_diag(grad_old, u)
        state.ene = ene
        state.epot = ene[state.instate]
        state.u = u
        state.nac = nac
        state.vk = self.vk_coupl_matrix(state.vel, nac)
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad_old_diag


    def mch_propagator(self, h_mch, vk, dt):
        h_total = np.diag(h_mch) - 1j*(vk) 
        ene, u = np.linalg.eigh(h_total)
        p_mch = np.linalg.multi_dot([u, np.diag(np.exp( -1j * ene * dt)), u.T.conj()])
        return p_mch

    def diag_propagator(self, u_new, dt, state):
        c_mch = state.ncoeff
        u = state.u
        ene = state.ene
        vk = state.vk
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch)
        c_diag = np.dot(u.T.conj(),c_mch)
        p_mch_new = self.mch_propagator(ene, vk, dt)
        p_diag_new = np.dot(u_new.T.conj() ,np.dot(p_mch_new,u))
        c_diag_new = np.dot(p_diag_new,c_diag) 
        return c_diag, c_diag_new, p_diag_new

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
    
    def probabilities(self, state, c_diag_dt, c_diag, p_diag_dt):
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
        return state

    def rescale_velocity(self, state, beta, alpha, diff, state_new, nac_av):
        if (beta**2 + 4*alpha*diff) < 0.0:
            """
            If this condition is satisfied, there is not hopping and 
            then the nuclear velocity simply are reversed.
            """
            gama_ji = beta/alpha
            state = self.new_velocity(state, gama_ji, nac_av)
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
                state = self.new_velocity(state, gama_ji, nac_av)
            else:
                gama_ji = (beta - np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                state = self.new_velocity(state, gama_ji, nac_av)
        return state

    def surface_hopping(self, state, nac_new, probs, ene_new):            
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
            state = self.rescale_velocity(state, beta, alpha, diff, state_new, nac_av) 
        return state, aleatory, acc_probs[state.instate]

    def new_surface(self, state, crd_new, t, dt):
        ene_new, u_new, nac_new, grad_new = self.get_ene_nac_grad(crd_new)
        grad_new_diag = self.grad_diag(grad_new, u_new)
        c_diag, c_diag_new, p_diag_new = self.diag_propagator(u_new, dt, state)
        probs = self.probabilities(state, c_diag_new, c_diag, p_diag_new)
        state, aleatory, acc_probs = self.surface_hopping(state, nac_new, probs, ene_new)
        print_var(t, dt, aleatory, acc_probs, state) #printing variables 
        state.ene = ene_new
        state.epot = ene_new[state.instate]
        state.u = u_new
        state.ncoeff = np.dot(state.u, c_diag_new)
        state.nac = nac_new
        state.vk = self.vk_coupl_matrix(state.vel, state.nac)
        state.ekin = self.cal_ekin(state.mass, state.vel)
        return grad_new_diag

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
    """
    
    def __init__(self, crd, vel, mass, instate, nstates, states, ncoeff):
        self.crd = crd
        self.vel = vel
        self.mass = mass
        self.instate = instate
        self.nstates = nstates
        self.states = states
        self.ncoeff = ncoeff
        self.ekin = 0
        self.epot = 0
        self.nac = {}
        self.ene = []
        self.vk = []
        self.u = []

    @classmethod
    def from_config(cls, config):
        crd = config['crd']
        vel = config['vel']
        mass = config['mass']
        instate = config['instate']
        nstates = config['nstates']
        states = config['states']
        ncoeff = config['ncoeff']
        return cls(crd, vel, mass, instate, nstates, states, ncoeff) 

    @classmethod
    def from_initial(cls, crd, vel, mass, instate, nstates, states, ncoeff):
        return cls(crd, vel, mass, instate, nstates, states, ncoeff)

def print_head():
    dash = '-' * 141
    print(dash)
    Header = ["MD_steps", "Time", "Position", "Velocity", "E_kinetic",\
                 "E_potential", "E_total", "Hopping_P", "Random", "State"]
    print(f"{Header[0]:>10s} {Header[1]:>10s} {Header[2]:>14s} {Header[3]:>14s}"\
                f"{Header[4]:>15s} {Header[5]:>17s} {Header[6]:>13s} {Header[7]:>15s} {Header[8]:>11s}  {Header[9]:>11s}")
    print(dash)

def print_var(t, dt, r, acc_probs, State):
    Var = (int(t/dt)+1,t,State.crd,State.vel,State.ekin,State.epot,State.ekin + State.epot,acc_probs,r,State.instate)
    print(f"{Var[0]:>8.0f} {Var[1]:>12.2f} {Var[2]:>14.2f}"\
                f"{Var[3]:>14.2f} {Var[4]:>15.3f} {Var[5]:>17.3f} {Var[6]:>13.3f} {Var[7]:>15.5f} {Var[8]:>11.5f} {Var[9]:>11.0f}")

def print_bottom():
    dash = '-' * 141
    print(dash)

if __name__=="__main__":
    elec_state = State.from_questions(config = "model_tully_1.ini")
    DY = VelocityVerletPropagator(0, elec_state)    
    try:
        result_2 = DY.run()
    except SystemExit as err:
        print("An error", err)
 
