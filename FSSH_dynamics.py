import numpy as np
import random

from abc import abstractmethod
#from verlet_2 import Verlet_Integrator
#from read_inp import Read_inp
#from surface_hopping import Surface_Hopping
from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
#from plots_one import Results
from pysurf.spp import ModelBase, SurfacePointProvider

class FSSH_dynamics:
    
    def __init__(self, mass, t, x_0, v_0, instate, nstates, states, ncoeff):
        self.mass = mass
        self.t = t
        self.x_0 = x_0
        self.v_0 = v_0
        self.instate = instate
        self.nstates = nstates
        self.states = states
        self.ncoeff = ncoeff
        self.dt = np.abs(0.05/v_0)
        self.t_max = 2.0*np.abs(self.x_0/self.v_0)
        self.md_step = int(self.t_max / self.dt)
        self.State = State(self.x_0, self.v_0, self.mass, self.instate, self.nstates, self.states, self.ncoeff)
        self.electronic = SurfaceHopping(self.nstates)
        #self.electronic = BornOppenheimer(self.nstates)
    
    def run(self):
        
        if self.t > self.t_max:
            raise SystemExit("Noting to be done")
        
        State = self.State
        
        ene, u, nac, grad_old = self.get_ene_nac_grad(State.crd)

        grad_old_diag = self.grad_diag(grad_old, u)
        acce_old = aceleration(grad_old_diag, State)
        State.epot = ene[State.instate]
        vk = self.vk_coupl_matrix(State.vel, nac)
       
        print_head()
        while(self.t <= self.t_max):
            """updating coordinates"""
            crd_new = position(State, acce_old, self.dt)
            """computing coefficients and hopping probabilities"""    
            ene_new, u_new, nac_new, grad_new = self.get_ene_nac_grad(crd_new)
            c_diag, c_diag_new, p_diag_new = self.diag_propagator(u, u_new, ene, vk, State) 
            probs = self.probabilities(State, c_diag_new, c_diag, p_diag_new) 
            State, acce_new, r, acc_probs = self.new_surface(State, grad_new, nac, nac_new, u_new, probs, ene_new)
            """updating velocities"""
            vel_new = velocity(State, acce_old, acce_new, self.dt)
            print_var(self.t, self.dt, r, acc_probs, State)
            """updating variables in State"""
            State = update_state(State, ene_new, crd_new, vel_new, u_new, c_diag_new) 
            vk = self.vk_coupl_matrix(State.vel, nac_new) 
            acce_old, u, ene, nac = acce_new, u_new, ene_new, nac_new 
            self.t += self.dt 
        return State, print_bottom()
    
    def population(self, State):
        rho = np.zeros(self.nstates)
        for i in range(self.nstates):
            rho[i] = np.dot(State.ncoeff[i],State.ncoeff.conj()[i]).real
        return rho

    def vk_coupl_matrix(self, vel, nac):
        vk = np.zeros((self.nstates,self.nstates))
        if np.isscalar(vel):
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if i < j:
                        vk[i,j] = vel*nac[i,j]
                        vk[j,i] = -vk[i,j]
            return vk
        else:
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if i < j:
                        vk[i,j] = np.dot(vel.flatten(),nac[i,j].flatten())
                        vk[j,i] = -vk[i,j]
            return vk
          
    def grad_diag(self, g_mch, u):
        g_diag = {}
        for i in range(self.nstates):
            g_diag.update({i:np.dot(u.T.conj()[i,:],u[:,i]).real*g_mch[i]})
        return g_diag

    def get_ene_nac_grad(self, crd):
        h_mch = self.electronic.get_energy(crd)
        grad_old = self.electronic.get_gradient(crd)
        nac = self.electronic.get_coupling(crd)
        ene, u = np.linalg.eigh(np.diag(h_mch))
        return ene, u, nac, grad_old

    def mch_propagator(self, h_mch, vk):
        h_total = np.diag(h_mch) - 1j*(vk) 
        ene, u = np.linalg.eigh(h_total)
        p_mch = np.linalg.multi_dot([u, np.diag(np.exp( -1j * ene * self.dt)), u.T.conj()])
        return p_mch

    def diag_propagator(self, u, u_new, h_mch, vk,  State):
        c_mch = State.ncoeff
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch)
        c_diag = np.dot(u.T.conj(),c_mch)
        p_mch_new = self.mch_propagator(h_mch, vk)
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
    
    def probabilities(self, State, c_diag_dt, c_diag, p_diag_dt):
        probs = np.zeros(self.nstates)
        instate = State.instate
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
        return ene_new[state_new] - ene_new[state_old]
    
    def beta_ji(self, vel, nac_av):
        if np.isscalar(vel):
            return vel*nac_av
        else:
            return np.dot(vel.flatten(),nac_av.flatten())
    
    def alpha_ji(self, nac_av):
        if np.isscalar(self.mass):
            return 0.5*(nac_av ** 2 )/self.mass
        else:
            alpha = 0.0
            for i, m in enumerate(self.mass):
                alpha += np.dot(nac_av[i], nac_av[i])/m
            return 0.5*alpha

    def new_surface(self, State, grad_new, nac_old, nac_new, u_new, probs, ene_new):            
        r = random.uniform(0,1)
        acc_probs = np.cumsum(probs)
        hopps = np.less(r, acc_probs)
        instate = State.instate
        if any(hopps): 
            for i in range(self.nstates):
                if hopps[i]:
                    state_to = State.states[i]
                    break
            nac_av = self.nac_average(instate,state_to,nac_old,nac_new)
            diff = self.diff_ji(instate,state_to,ene_new)
            beta = self.beta_ji(State.vel, nac_av)
            alpha = self.alpha_ji(nac_av)
            if (beta**2 - 4*alpha*diff) < 0.0:
                """
                If this condition is satisfied, there is not hopping and 
                then the nuclear velocity simply are reversed.
                """
                g_diag_dt = self.grad_diag(grad_new, u_new)
                a_dt = aceleration(g_diag_dt, State)
                gama_ji = beta/alpha
                if np.isscalar(State.vel) and np.isscalar(self.mass):
                    State.vel = State.vel - gama_ji*(nac_av/self.mass)
                else:
                    for i, m in enumerate(self.mass):
                        State.vel[i] = State.vel[i] - gama_ji*(nac_av[i]/m)
            else:
                """
                If this condition is satisfied, a hopping from 
                current state to the first true state takes place 
                and the current nuclear velocity is ajusted in order 
                to preserve the total energy.
                """
                State.instate = state_to
                g_diag_dt = self.grad_diag(grad_new, u_new)
                a_dt = aceleration(g_diag_dt, State)
                if (beta < 0.0):
                    gama_ji = (beta + np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                    if np.isscalar(State.vel) and np.isscalar(self.mass):
                        State.vel = State.vel - gama_ji*(nac_av/self.mass)
                    else:
                        for i, m in enumerate(self.mass):
                            State.vel[i] = State.vel[i] - gama_ji*(nac_av[i]/m)
                else:
                    gama_ji = (beta - np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)

                    if np.isscalar(State.vel) and np.isscalar(self.mass):
                        State.vel = State.vel - gama_ji*(nac_av/self.mass)
                    else:
                        for i, m in enumerate(self.mass):
                            State.vel[i] = State.vel[i] - gama_ji*(nac_av[i]/m)
        else:
            g_diag_dt = self.grad_diag(grad_new, u_new)
            a_dt = aceleration(g_diag_dt, State)
        return State, a_dt, r, acc_probs[State.instate]

class Electronic:

    needed_properties = ["energy","gradient","coupling"]

    def __init__(self, nstates):
        self.nstates = nstates
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

class SurfaceHopping(Electronic):

    needed_properties = ["energy", "gradient", "coupling"]

    def __init__(self, nstates):
        self.nstates = nstates
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

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

class BornOppenheimer(Electronic):

    needed_properties = ["energy","gradient","coupling"]

    def __init__(self, nstates):
        self.nstates = nstates
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

class State:
    
    def __init__(self, crd, vel, mass, instate, nstates, states, ncoeff):
        self.crd = crd
        self.vel = vel
        self.mass = mass
        self.instate = instate
        self.nstates = nstates
        self.states = states
        self.ncoeff = ncoeff
        self.ekin = cal_ekin(mass, vel)
        self.epot = 0
    
    @classmethod
    def from_initial(cls, crd, vel, mass, instate, nstates, states, ncoeff):
        ekin = cal_ekin(mass, vel)
        return cls(crd, vel, mass, instate, nstates, states, ncoeff, ekin, epot)

def cal_ekin(mass, vel):
    ekin = 0
    if np.isscalar(mass) and np.isscalar(vel):
        ekin = 0.5*mass*vel**2
    else:
        for i, m in enumerate(mass):
            ekin += 0.5*m*np.dot(vel[i],vel[i])
    return ekin

def update_state(State, ene_new, crd_new, vel_new, u_new, c_diag_new):
    State.crd = crd_new
    State.vel = vel_new
    State.ekin = cal_ekin(State.mass, vel_new)
    State.epot = ene_new[State.instate]
    State.ncoeff = np.dot(u_new, c_diag_new)
    return State

def aceleration(grad, State):
    return -grad[State.instate]/State.mass

def position(State, a_0, dt):
    return State.crd + State.vel*dt + 0.5*a_0*dt**2

def velocity(State, a_0, a_1, dt):
    return State.vel + 0.5*(a_0 + a_1)*dt
 

def print_head():
    dash = '-' * 141
    print(dash)
    Header = ["MD_steps", "Time", "Position", "Velocity", "E_kinetic",\
                 "E_potential", "E_Total", "Hopping_P", "Random", "State"]
    print(f"{Header[0]:>10s} {Header[1]:>10s} {Header[2]:>14s} {Header[3]:>14s}"\
                f"{Header[4]:>15s} {Header[5]:>17s} {Header[6]:>13s} {Header[7]:>15s} {Header[8]:>11s}  {Header[9]:>11s}")
    print(dash)

def print_var(t, dt, r, acc_probs, State):
    Var = (int(t/dt)+1,t,State.crd,State.vel,State.ekin,State.epot,State.ekin + State.epot,acc_probs,r,State.instate)
    print(f"{Var[0]:>10.0f} {Var[1]:>10.3f} {Var[2]:>14.4f}"\
                f"{Var[3]:>14.4f} {Var[4]:>15.4f} {Var[5]:>17.4f} {Var[6]:>13.4f} {Var[7]:>15.4f} {Var[8]:>11.4f} {Var[9]:>11.0f}")

def print_bottom():
    dash = '-' * 141
    print(dash)

if __name__=="__main__":
#    HO = Electronic(0,2)
#    result = HO.get_gradient(-4)
#    print(result)

#    ST = State(4,5,5,0,2)
#    print(ST.crd,ST.mass,ST.vel,ST.instate)
#
    states = (0,1)
    wstates = (1,0)   
    DY = FSSH_dynamics(1, 0, -4, 1, 0, 2, states, wstates)
    try:
        result_2 = DY.run()
    except SystemExit as err:
        print("An error", err)
 
