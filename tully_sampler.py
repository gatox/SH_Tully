import numpy.random as random

from pysurf.sampling.base_sampler import DynSamplerBase



class TullySampler(DynSamplerBase):
    
    _questions = """ 
    # position or moment must be different from zero 
    crd = -4 :: float
    mass =  2000.0 :: float
    momentum = 2 :: float
    model = 1 :: int
    state = 0 :: int
    """
    


    @classmethod
    def from_config(cls, config):
        return cls(config['crd'], config['momentum'], config['mass'], config['model'], config['state'])
        
    def __init__(self, crd, momentum, mass, model, state):
        """Initialize a new Tully Sampling"""
        
        self.crd = crd
        self.moment = momentum
        self.mass = mass
        self.model = model
        self.state = state
    
    def get_condition(self):
        
        """Initial sampling condition according to
           John C. Tully, J. Chem. Phys. 93, 1061 (1990):
           
               Initial state: the lowest energy state; 0
               Position: asymtotic negative region; [-10,0[
               Momentum: positive momentum; [0,30]    
       
       """
        
        if self.model < 3:
            self.crd = random.random()*(-10.0)
        else:
            self.crd = random.random()*(-15.0)
            
        self.momentum = random.random()*(30.0)
        return DynSamplerBase.condition(self.crd, self.mass*self.momentum, self.state)
 

if __name__=="__main__":
    HO = TullySampler(2, 0, 2000, 3, 0)
    print(HO.get_condition())
    
#a = TullySampler.from_questions(config = "values")
#print(a.get_condition())