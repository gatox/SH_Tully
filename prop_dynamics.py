import numpy as np

from pysurf.database import PySurfDB
from pysurf.colt import Colt

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
    rescale_vel = momentum :: str :: momentum, nacs 
    coupling = nacs :: str :: nacs, wf_overlap
    method = Surface_Hopping :: str :: Surface_Hopping, Born_Oppenheimer  
    """

    def __init__(self, crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, rescale_vel, coupling, method, atomids):
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
        self.rescale_vel = rescale_vel
        self.coupling = coupling
        if self.rescale_vel == "nacs" and self.coupling != "nacs":
            raise SystemExit("Wrong coupling method or wrong rescaling velocity approach")
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
        rescale_vel = config['rescale_vel']
        coupling = config['coupling']
        method = config['method']
        return cls(crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, rescale_vel, coupling, method, atomids)  

    @staticmethod
    def read_db(db_file):
        db = PySurfDB.load_database(db_file, read_only=True)
        crd = np.copy(db['crd'][0])
        vel = np.copy(db['veloc'][0])
        atomids = np.copy(db['atomids'])
        mass = np.copy(db['masses'])
        return crd, vel, mass, atomids

    @classmethod
    def from_initial(cls, crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, rescale_vel, coupling, method):
        return cls(crd, vel, mass, t, dt, mdsteps, instate, nstates, states, ncoeff, prob, rescale_vel, coupling, method)

if __name__=="__main__":
    State.from_questions(config = "prop.inp")
