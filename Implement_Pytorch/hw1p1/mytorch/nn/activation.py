import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
       self.A =  1/(1+ np.exp(-Z))
       return self.A
    
    def backward(self):
        dAdZ = self.A  - self.A*self.A
        return dAdZ



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A =  (np.exp(Z)- np.exp(-Z))/(np.exp(Z)+ np.exp(-Z))
        return self.A
    
    def backward(self):
        dAdZ = 1  - self.A*self.A
        return dAdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        
        self.A = np.maximum(Z, np.zeros(Z.shape))
        return self.A
    
    def backward(self):
        dAdZ = np.where(self.A >0, 1.0, 0.0)
        return dAdZ
