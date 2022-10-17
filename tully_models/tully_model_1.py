import numpy as np
from collections import namedtuple

from pysurf.spp import ModelBase, SurfacePointProvider
from pysurf.system import Mode
from colt import Colt


class Tully_1(ModelBase):
#class Tully_1(Colt):

    _questions = """ 
    # chosen parameters
    a = 0.01 :: float
    b = 1.6 :: float
    c = 0.005 :: float
    d = 1.0  :: float
    """
    
    implemented = ["energy", "gradient", "di_energy", "di_gradient", "coupling"]
    
    def __init__(self, a,b,c,d):
        """ Tully model 1: Simple avoided crossing 

            Diabatic potential, parameters:
            ------------------------------
                a: float
                b: float
                c: float
                d: float
        """
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    def from_config(cls,config):
        return cls(config['a'], config['b'], config['c'], config['d'])

    
    def _di_energy(self, x):
        """ returns the matrix of diabatic energies at position x """
        sign_a  = np.copysign(self.a,x)
        V_11 = sign_a*(1-np.exp(-self.b*np.abs(x)))
        V_22 = -V_11
        V_12 = self.c*np.exp(-self.d*(x**2))
        return np.array([ [V_11, V_12], [V_12, V_22] ])

    
 
    def _di_gradient(self, x):
        """ returns the matrix of gradients of diabatic energies at position x """
        D_11 = self.a*self.b*np.exp(-self.b*np.abs(x))
        D_22 = -D_11
        D_12 = -self.d*(2*x)*self.c*np.exp(-self.d*(x**2))
        return np.array([ [D_11, D_12], [D_12, D_22] ])
    
    def _energy(self, x):
        """ returns the matrix of adiabatic energies at position x """
        u_ij_di = self._di_energy(x)
        energies, coeff = np.linalg.eigh(u_ij_di)
        return energies
    
    def _gradient(self, x): 
        """ returns the matrix of adiabatic energies at position x """
        u_ij_di = self._di_energy(x)
        DV = self._di_gradient(x)
        energies, coeff = np.linalg.eigh(u_ij_di)

        gradients= np.dot(coeff.T, np.dot(DV, coeff))
        grad = np.diag(gradients)
        return {0: grad[0], 1:grad[1]}

    
    def _n_coupling(self, x):
        """ returns the couplings at position x using np.linalg.eigh function. 

            In Tully's paper, the Hamiltonian is expressed in a diabatic basis. 
            If the diabatic basis is transformed into adiabatic basis, then 
            the Hellmann-Feynman theorem:
                
                <\phi_i|d/dx|\phi_j> = <\phi_i|dH/dx|\phi_j>/(e_j - e_i),
                
            can be used to compute the nonadiabatic coupling strength (since, 
            this theorem holds for the adiabatic basis). 
            
            The diabatic basis (\psi_j) is related to the adiabatic basis (\phi_k) by 
            the unitary tranformation:
                
                \psi_j = \sum_k \phi_k U_{k,j}
                
        """
        u_ij_di = self._di_energy(x)
        DV = self._di_gradient(x)
        energies, coeff = np.linalg.eigh(u_ij_di)

        dij= np.dot(coeff[:,0].T, np.dot(DV, coeff[:,1]))
        dE = energies[1] - energies[0]
        return np.array([ [0, -dij/(50*dE)], [dij/(50*dE), 0 ] ])
    
    def _a_coupling(self, x):
        """ returns the couplings at position x using the explicit expression
        for the eigenvalues and eigenvectors of a matrix 2x2"""
        H = self._di_energy(x)
        DH = self._di_gradient(x)
        dE = np.sqrt( (H[0,0] - H[1,1])**2 + 4*H[0,1]**2 )
        ei = 0.5*( (H[0,0] + H[1,1]) - dE )
        ej = 0.5*( (H[0,0] + H[1,1]) + dE )
        
        #coefficients with ei (the lowest energy)
        ci_1 = (H[0,1]/np.sqrt( H[0,1]**2 + (H[0,0]-ei)**2 ))
        ci_2 = -(H[0,0]-ei)/np.sqrt( H[0,1]**2 + (H[0,0]-ei)**2 )
        #coefficients with ej
        cj_1 = (H[0,1]/np.sqrt( H[0,1]**2 + (H[0,0]-ej)**2 ))
        cj_2 = -(H[0,0]-ej)/np.sqrt( H[0,1]**2 + (H[0,0]-ej)**2 )
    
        coeff = np.array([ [ci_1, cj_1], [ci_2, cj_2] ])

        dij= np.dot(coeff[:,0].T, np.dot(DH, coeff[:,1]))
        return np.array([ [0, dij/(50*dE)], [-dij/(50*dE), 0 ] ])
    
    def get(self, request):
        crd = request.x
        for prop in request:
            if prop == 'energy':
                request.set('energy', self._energy(crd))
            if prop == 'gradient':
                request.set('gradient', self._gradient(crd))
            if prop == 'di_energy':
                request.set('di_energy', self._di_energy(crd))
            if prop == 'di_gradient':
                request.set('di_gradient', self._di_gradient(crd))
            if prop == 'n_coupling':
                request.set('n_coupling', self._n_coupling(crd))
            if prop == 'a_coupling':
                request.set('a_coupling', self._a_coupling(crd))
        return request


#if __name__=="__main__":
#    HO = Tully_1(a = 0.01, b = 1.6, c = 0.005, d = 1.0)
#    request = namedtuple('request', 'crd energys gradients coupling')
#    request.crd = np.array(1)
#    print(HO._di_energy(request.crd))
#    print(HO._di_gradient(request.crd))
#    print(HO._n_coupling(request.crd))
#    print(HO._a_coupling(request.crd))
    
a = SurfacePointProvider.from_questions(["energy"], 2, 1, config ="Tully1.in")

#a = Tully_1.from_questions(config = "values")
#print("Diabatic energy: ", a._di_energy(-10))
#print("Gradient: ", a._di_gradient(-10))
#print("Coupling: ", a._n_coupling(-10))
#print("Coupling: ", a._a_coupling(-10))
