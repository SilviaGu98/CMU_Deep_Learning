import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        # TODO
        self.upsample1d = Upsample1d(upsampling_factor)  # TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                 weight_init_fn, bias_init_fn)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)  # TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO

        # Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample1d.backward(delta_out)  # TODO

        return dLdA


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                 weight_init_fn, bias_init_fn)  # TODO
        self.upsample2d = Upsample2d(upsampling_factor)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)  # TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample2d.backward(delta_out)  # TODO

        return dLdA
