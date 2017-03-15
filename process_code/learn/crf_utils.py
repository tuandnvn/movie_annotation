import numpy as np
import tensorflow as tf

def gather_2d(params, indices):
    # only for two dim now
    shape = params.get_shape().as_list()
    assert len(shape) == 2, 'only support 2d matrix'
    indices_shape = indices.get_shape().as_list()
    assert indices_shape[1] == 2, 'only support indexing on both dimensions'
    
    flat = tf.reshape(params, [shape[0] * shape[1]])
#     flat_idx = tf.slice(indices, [0,0], [shape[0],1]) * shape[1] + tf.slice(indices, [0,1], [shape[0],1])
#     flat_idx = tf.reshape(flat_idx, [flat_idx.get_shape().as_list()[0]])
    
    flat_idx = indices[:,0] * shape[1] + indices[:,1]
    return tf.gather(flat, flat_idx)

def gather_2d_to_shape(params, indices, output_shape):
    flat = gather_2d(params, indices)
    return tf.reshape(flat, output_shape)

# x -> (x, size)
def expand( tensor, size, axis = 1 ):
    return tf.stack([tensor for _ in xrange(size)], axis = axis)

# x -> (size, x)
def expand_first( tensor, size ):
    return tf.stack([tensor for _ in xrange(size)])

