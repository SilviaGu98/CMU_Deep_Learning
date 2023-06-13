import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]  # TODO
        self.M = (np.sum(Z, axis = 0)/self.N).reshape(1,-1)  # TODO
        
        self.V = (np.sum((Z - self.M)**2, axis = 0)/self.N).reshape(1,-1)  # TODO
        
        self.Ones = np.ones((self.N, 1))

        if eval == False:
            # training mode
            
            self.NZ = (self.Z - self.Ones.dot(self.M))/(self.Ones.dot(np.sqrt(self.V+self.eps)))  # TODO
            self.BZ = self.Ones.dot(self.BW)*self.NZ + self.Ones.dot(self.Bb)  # TODO

            self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M   # TODO
            self.running_V = self.alpha*self.running_V + (1-self.alpha)*self.V  # TODO
        else:
            # inference mode
            self.NZ = (self.Z - self.Ones.dot(self.running_M))/(self.Ones.dot(np.sqrt(self.running_V+self.eps)))  # TODO
            self.BZ = self.Ones.dot(self.BW)*self.NZ + self.Ones.dot(self.Bb)  # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ*self.NZ, axis = 0)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis = 0)  # TODO

        dLdNZ = dLdBZ*self.BW  # TODO #element wise

        dLdV = -0.5*np.sum(dLdNZ*(self.Z - self.M)*((self.V + self.eps)**(-3/2)),axis = 0) # TODO

        first_term_dmu = -(np.sum(dLdNZ*((self.V + self.eps)**(-1/2)), axis=0))
        second_term_dmu = - (2 / self.N) * (dLdV) * (np.sum(self.Z - self.M, axis=0))
        dLdM = first_term_dmu + second_term_dmu # TODO
       
        first_term_dx = dLdNZ*((self.V + self.eps)**(-1/2))
        second_term_dx = dLdV * (2/self.N) * (self.Z - self.M)
        third_term_dx = dLdM * (1/self.N)

        dLdZ = first_term_dx + second_term_dx + third_term_dx  # TODO

        return dLdZ
