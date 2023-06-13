import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        output_height = A.shape[2]-self.kernel_size+1
        output_width = A.shape[3]-self.kernel_size+1
        Z = np.zeros((A.shape[0], self.out_channels, output_height, output_width)) # TODO
        for n in range(Z.shape[0]):
            for c in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        Z[n,c,h,w] = np.sum(A[n,:,h:h+self.kernel_size,w:w+self.kernel_size]*self.W[c,:,:]) +self.b[c]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        for o in range(self.out_channels):
            for i in range(self.in_channels):
                for h in range(self.dLdW.shape[2]):
                    for w in range(self.dLdW.shape[2]):
                    
                        self.dLdW[o,i,h,w] += np.sum(self.A[:,i,h:h+dLdZ.shape[2],w:w+dLdZ.shape[3]]*dLdZ[:,o,:,:])
                   
        # self.dLdW = None  # TODO
        self.dLdb = np.sum(dLdZ, axis = (0,2,3)) # each output channel has a bias
        dLdA = np.zeros(self.A.shape) # TODO
        flipped_W = np.flip(self.W, axis=(2,3))
        padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1)), 'constant') 
        for n in range(dLdA.shape[0]):
            for c in range(self.in_channels):
                for h in range(dLdA.shape[2]):
                    for w in range(dLdA.shape[3]):
                        dLdA[n,c,h,w] += np.sum(padded_dLdZ[n,:,h:h+self.kernel_size,w:w+self.kernel_size]*flipped_W[:,c,:,:])

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        # TODO
        con_A = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(con_A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        # TODO
        down_dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(down_dLdZ)  # TODO

        return dLdA
