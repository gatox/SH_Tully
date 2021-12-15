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
        #self.electronic = SurfaceHopping(self.nstates)
        self.electronic = BornOppenheimer(self.nstates)
        self.State = State(self.x_0, self.v_0, self.mass, self.instate, self.nstates, self.states, self.ncoeff)
    
    def run(self):
        
        if self.t > self.t_max:
            raise SystemExit("Noting to be done")
        
        State = self.State
        electronic = self.electronic
        
        ene, u, nac, grad_old = self.get_ene_nac_grad_diag(State.crd) #Verlet: diag. at x_0
        State.epot = ene[State.instate]
        vk = self.vk_coupl_matrix(State.vel, nac)
       
        print_head()
        while(self.t <= self.t_max):
            crd_new = update_coordinates(State, grad_old_diag[State.instate], self.dt)
            ene_new, u_new, nac_new, grad_new_diag = self.get_ene_nac_grad_diag(crd_new) #Verlet: diag. at x_dt
            c_diag, c_diag_new = self.diag_propagator(u, u_new, ene, vk,  State) 
            

            vel_new = update_velocities(State, grad_old_diag[State.instate], grad_new_diag[State.instate], self.dt)
            print_var(self.t, State)
            State, grad_dia_old = update_state(State, crd_new, vel_new), grad_dia_new
           
            self.t += self.dt 
        return State
    
    def rho_matrix(self, State):
        rho = np.zeros(self.nstates)
        for i in range(self.nstates):
            rho[i] = np.dot(State.ncoeff[i],State.ncoeff.conj()[i]).real
        return np.diag(rho)

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
          
    def grad_matrix(self, g_mch, ene, nac, u):
        g_diag = {}
        for i in range(self.nstates):
            for j in range(self.nstates):
                if i < j:
                    prod = (u.T.conj()[i][j]*u[i][j]).real*(ene[j]- ene[i])*nac[i,j]
                    g_diag.update({(i,j): prod})
                    g_diag.update({(j,i):-prod})
                elif i == j:
                    g_diag.update({(i,j):(u.T.conj()[i][j]*u[i][j]).real*g_mch[i]})
        return g_diag

    def get_ene_nac_grad_diag(self, crd):
        h_mch = self.electronic.get_energy(crd)
        grad_old = self.electronic.get_gradient(crd)
        nac = self.electronic.get_coupling(crd)
        ene, u = np.linalg.eigh(h_mch)
        grad_old_diag = self.grad_matrix(grad_old, ene, nac, u)
        return ene, u, nac, grad_old_diag

    def mch_propagator(self, h_mch, vk):
        h_total = np.diag(h_mch) - 1j*(vk) 
        ene, u = np.linalg.eigh(h_total)
        p_mch = np.linalg.multi_dot([u, np.diag(np.exp( -1j * np.diag(ene) * self.dt)), u.T.conj()])
        return p_mch

    def diag_propagator(self, u, u_new, h_mch, vk,  State):
        c_mch = State.ncoeff
        if isinstance(c_mch, np.ndarray) != True:
            c_mch = np.array(c_mch)
        c_diag = np.dot(u.T.conj(),c_mch)
        p_mch_new = self.mch_propagator(h_mch, vk)
        p_diag_new = np.dot(u_new.T.conj() ,np.dot(p_mch_new,u))
        c_diag_new = np.dot(p_diag_new,c_diag) 
        return c_diag, c_diag_new

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
    
    def probabilities(self, instate, c_diag_dt, c_diag, p_diag_dt):
        probs = np.zeros(self.nstates)
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
            beta = 0.0
            dim = len(self.mass)  
            for i in range(dim):
                beta +=  np.dot(vel[i],nac_av[i])
            return beta
    
    def alpha_ji(self, nac_av):
        if np.isscalar(self.mass):
            return 0.5*(nac_av ** 2 )/self.mass
        else:
            alpha = 0.0
            dim = len(self.mass)
            for i in range(dim):
                alpha += np.dot(nac_av[i], nac_av[i])/self.mass[i]
            return 0.5*alpha

    def new_surface(self, G_dt, State, v_0, Ene_dt, nac_old, nac_new, U_dt, probs, ene_new):
            
        r = random.uniform(0, 1)
        acc_probs = np.cumsum(probs)
        hopps = np.less(r, acc_probs)
        instate = State.instate
        if any(hopps): 
            for i in range(self.nstates):
                if hopps[i]:
                    state_to = States.states[i]
                    break
            nac_av = self.nac_average(instate,state_to,nac_old,nac_new)
            diff = self.diff_ji(instate,state_to,ene_new)
            beta = self.beta_ji(State.vel, nac_av)
            alpha = self.alpha_ji(nac_av)
            if (beta**2 - 4*alpha*diff) < 0:
                """
                If this condition is satisfied, there is not hopping and 
                then the nuclear velocity simply are reversed.
                """
                G_diag_dt = self.gradient(G_dt, state, Ene_dt, NAC_dt, U_dt)
                a_dt = aceleration(G_diag_dt, self.mass)
                gama_ji = beta/alpha
                v_0 = v_0 - gama_ji*(NAC_av/self.mass)
            else:
                """
                If this condition is satisfied, a hopping from 
                current state to the first true state takes place 
                and the current nuclear velocity is ajusted in order 
                to preserve the total energy
                """
                state = state_to
                G_diag_dt = self.gradient(G_dt, state, Ene_dt, NAC_dt, U_dt)
                a_dt = aceleration(G_diag_dt, self.mass)
                if (beta < 0.0):
                    gama_ji = (beta + np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                    v_0 = v_0 - gama_ji*(NAC_av/self.mass)
                else:
                    gama_ji = (beta - np.sqrt(beta**2 + 4*alpha*diff))/(2*alpha)
                    v_0 = v_0 - gama_ji*(NAC_av/self.mass)
        else:
            G_diag_dt = self.gradient(G_dt, state, Ene_dt, NAC_dt, U_dt)
            a_dt = aceleration(G_diag_dt, self.mass)
        return state, v_0, a_dt, r, acc_probs[state]

class Electronic:

    needed_properties = ["energy","gradient","coupling"]

    def __init__(self, nstates):
        self.nstates = nstates
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

    @abstractmethod
    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def rescale_velocity(self, state):
        pass

    def get_energy(self, crd):
        result = self.spp.request(crd, ['energy'])
        return result['energy']

    def get_coupling(self, crd):
        result = self.spp.request(crd, ['coupling'])
        return result['coupling']

class SurfaceHopping(Electronic):

    needed_properties = ["energy", "gradient", "coupling"]

    def __init__(self, instate, nstates, ncoeff):
        self.instate = instate
        self.nstates = nstates
        self.ncoeff = ncoeff
        self.spp = SurfacePointProvider.from_questions(["energy","gradient","coupling"], self.nstates, 1, config ="FSSH.in")

    def get_gradient(self, crd):
        result = self.spp.request(crd, ['gradient'])
        return result['gradient']

    def rescale_velocity(self, state):
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
        self.electronic = Electronic(self.nstates)
        self.epot = 0
        self.etot = self.ekin + self.epot
    
    @classmethod
    def from_initial(cls, crd, vel, mass, instate, nstates, states, ncoeff):
        ekin = cal_ekin(mass, vel)
        epot = 0
        etot = ekin + epot
        return cls(crd, vel, mass, instate, nstates, states, ncoeff, ekin, epot, etot)

def cal_ekin(mass, vel):
    ekin = 0
    if np.isscalar(mass) and np.isscalar(vel):
        ekin = 0.5*mass*vel**2
    else:
        for i, m in enumerate(mass):
            ekin += 0.5*m*np.dot(vel[i],vel[i])
    return ekin


def update_coordinates(State, grad, dt):
    acce = aceleration(grad, State)
    crd_update = position(State.crd, State.vel, acce, dt)
    return crd_update

def update_velocities(State, grad_old, grad_new, dt):
    acce_old = aceleration(grad_old, State)
    acce_new = aceleration(grad_new, State)
    vel_update = velocity(State.vel, acce_old, acce_new, dt)
    return vel_update

def update_state(State, poten, crd_new, vel_new):
    State.crd = crd_new
    State.vel = vel_new
    State.ekin = cal_ekin(State.mass, vel_new)
    State.epot = poten
    State.etot = State.ekin + State.epot
    return State

def aceleration(grad, State):
    return -grad/State.mass

def position(x_0, v_0, a_0, dt):
    return x_0 + v_0*dt + 0.5*a_0*dt**2

def velocity(self, v_0, a_0, a_1, dt):
    return v_0 + 0.5*(a_0 + a_1)*dt
 

def print_head():
    dash = '-' * 81
    print(dash)
    Header = ["Time", "Position", "Velocity", "E_Kinetic", "E_Potential" ,\
          "E_Total", "State"]
    print(f"{Header[0]:>6s} {Header[1]:>10s} {Header[2]:>10s}"\
               f"{Header[3]:>13s} {Header[4]:>13s} {Header[5]:>13s} {Header[6]:>9s}")
    print(dash)

def print_var(t, State):
    Var = (t,State.crd,State.vel,State.ekin,State.epot,State.etot,State.instate)
    print(f"{Var[0]:>6.2f} {Var[1]:>10.3f} {Var[2]:>10.3f}"\
                f"{Var[3]:>13.3f} {Var[4]:>13.3f} {Var[5]:>13.3f} {Var[6]:>9.0f}")


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
 
