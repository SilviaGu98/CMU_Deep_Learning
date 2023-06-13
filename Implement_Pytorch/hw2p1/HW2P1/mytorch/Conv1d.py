# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        output_size = A.shape[2]-self.kernel_size+1
        Z = np.zeros((A.shape[0], self.out_channels, output_size)) # TODO
        for n in range(Z.shape[0]):
            for c in range(self.out_channels):
                for w in range(output_size):
                    Z[n,c,w] = np.sum(A[n,:,w:w+self.kernel_size]*self.W[c,:,:]) +self.b[c]
        # for n in range(Z.shape[0]):
        #     for w in range(output_size):
        #         Z[n,:,:] = np.tensordot(A[n,:,w:w+self.kernel_size], self.W, axes = ([1,2],[1,2])) + self.b
        # # Z = np.tensordot(A, self.W, axes = ([1,2],[1,2])) + self.b.reshape(1,-1,1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        
        for o in range(self.out_channels):
            for i in range(self.in_channels):
                for k in range(self.dLdW.shape[2]):
                    self.dLdW[o,i,k] += np.sum(self.A[:,i,k:k+dLdZ.shape[2]]*dLdZ[:,o,:])
                   
        # self.dLdW = None  # TODO
        self.dLdb = np.sum(dLdZ, axis = (0,2)) # each output channel has a bias
        dLdA = np.zeros(self.A.shape) # TODO
        flipped_W = np.flip(self.W, axis=2)
        padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1)), 'constant') 
        for n in range(dLdA.shape[0]):
            for c in range(self.in_channels):
                for w in range(dLdA.shape[2]):
                    dLdA[n,c,w] += np.sum(padded_dLdZ[n,:,w:w+self.kernel_size]*flipped_W[:,c,:])

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 =Conv1d_stride1(in_channels, out_channels, kernel_size,weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        con_A = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(con_A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        down_dLdZ = self.downsample1d.backward(dLdZ)  # TODO

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(down_dLdZ)  # TODO

        return dLdA
