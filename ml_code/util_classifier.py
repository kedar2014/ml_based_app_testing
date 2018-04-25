import numpy as np
from math import sqrt
import tensorflow as tf

class Utilities:

    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def calc_accuracy(self,correct_pedictions_batch,predictions_batch):
        self.total_predictions = self.total_predictions +  predictions_batch
        self.correct_predictions += np.sum(correct_pedictions_batch)
        accuracy = (self.correct_predictions / float(self.total_predictions))*100
        return accuracy , np.sum(correct_pedictions_batch)

    def weights_to_image_grid(self,kernel,pad = 1):
        '''Visualize conv. filters as an image (mostly for the 1st layer).
        Arranges filters into a grid, with some paddings between adjacent filters.
        Args:
            kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
            pad:               number of black pixels around each filter (between them)
        Return:
            Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
        '''
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))
        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad

        channels = kernel.get_shape()[2]

        # put NumKernels to the 1st dimension
        x = tf.transpose(x, (3, 0, 1, 2))
        # organize grid on Y axis
        x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x = tf.transpose(x, (0, 2, 1, 3))
        # organize grid on X axis
        x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x = tf.transpose(x, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x = tf.transpose(x, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x
