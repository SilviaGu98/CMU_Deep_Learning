import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.out_width = A.shape[2] - self.kernel + 1
        self.out_height = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], self.out_width, self.out_height)) 
        for n in range(A.shape[0]):
            for c in range(A.shape[1]): 
                for i in range(self.out_width):
                    for j in range(self.out_height):      
                        Z[n,c,i,j] = np.max(A[n,c,i:i+self.kernel,j:j+self.kernel])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        
        for n in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                for i in range(self.out_width):
                    for j in range(self.out_height):   
                        i_max, j_max = np.where(self.A[n,c,i:i+self.kernel,j:j+self.kernel] == np.max(self.A[n,c,i:i+self.kernel,j:j+self.kernel]))
                        
                        i_max, j_max = i_max[0], j_max[0] # take element from array
                        
                        dLdA[n,c,i:i+self.kernel,j:j+self.kernel][i_max,j_max] += dLdZ[n,c,i,j]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.out_width = A.shape[2] - self.kernel + 1
        self.out_height = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], self.out_width, self.out_height)) 
        for n in range(A.shape[0]):
            for c in range(A.shape[1]): 
                for i in range(self.out_width):
                    for j in range(self.out_height):      
                        Z[n,c,i,j] = np.mean(A[n,c,i:i+self.kernel,j:j+self.kernel])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        num_elements = self.kernel**2
        dLdA = np.zeros(self.A.shape)
        for n in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                for i in range(self.out_width):
                    for j in range(self.out_height):
                        dLdA[n,c,i:i+self.kernel,j:j+self.kernel] += dLdZ[n,c,i,j]/num_elements

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = None  # TODO
        self.downsample2d = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A = A
        self.out_width = (A.shape[2] - self.kernel)//self.stride + 1
        self.out_height = (A.shape[3] - self.kernel)//self.stride + 1
        Z = np.zeros((A.shape[0], A.shape[1], self.out_width, self.out_height)) 
        for n in range(A.shape[0]):
            for c in range(A.shape[1]): 
                for i in range(self.out_width):
                    for j in range(self.out_height):      
                        Z[n,c,i,j] = np.max(A[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        
        for n in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                for i in range(self.out_width):
                    for j in range(self.out_height):   
                        i_max, j_max = np.where(self.A[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel] 
                                                == np.max(self.A[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel]))
                        
                        i_max, j_max = i_max[0], j_max[0] # take element from array
                        
                        dLdA[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel][i_max,j_max] += dLdZ[n,c,i,j]

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = None  # TODO
        self.downsample2d = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.out_width = (A.shape[2] - self.kernel)//self.stride + 1
        self.out_height = (A.shape[3] - self.kernel)//self.stride + 1
        Z = np.zeros((A.shape[0], A.shape[1], self.out_width, self.out_height)) 
        for n in range(A.shape[0]):
            for c in range(A.shape[1]): 
                for i in range(self.out_width):
                    for j in range(self.out_height):      
                        Z[n,c,i,j] = np.mean(A[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        num_elements = self.kernel**2
        dLdA = np.zeros(self.A.shape)
        for n in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                for i in range(self.out_width):
                    for j in range(self.out_height):
                        dLdA[n,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel] += dLdZ[n,c,i,j]/num_elements

        return dLdA
