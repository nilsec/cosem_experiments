from funlib.learn.tensorflow import models
import tensorflow as tf
import os
import json
from micron.network import max_detection
import numpy as np

def create_network(input_shape, 
                   name, 
                   setup,
                   voxel_size,
                   nms_window,
                   nms_threshold=0.5,
                   double_suppression_size=3,
                   add_noise=False):

    tf.reset_default_graph()

    soft_mask = tf.placeholder(tf.float32, shape=input_shape)
    soft_mask_batched = tf.reshape(soft_mask, (1, 1) + input_shape)

    pred_maxima, pred_reduced_maxima = max_detection(tf.reshape(soft_mask, [1] + soft_mask.get_shape().as_list() + [1]), 
                                                                nms_window, nms_threshold, double_suppression_size, add_noise)

    output_shape_batched = soft_mask_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    print("pred maxima", pred_maxima.get_shape().as_list())

    output_shape = output_shape[1:]
    print("input shape : %s"%(input_shape,))
    print("output shape: %s"%(output_shape,))

    tf.train.export_meta_graph(filename=name + '.meta')

    config = {
        'soft_mask': soft_mask.name,
        'pred_maxima': pred_maxima.name,
        'pred_reduced_maxima': pred_reduced_maxima.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
    }

    config['outputs'] = {
            'reduced_maxima':
                {"out_dims": 1,
                    "out_dtype": "uint8"}
            }

    config['voxel_size'] = voxel_size

    with open(name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    z = 47
    xy = 486
    create_network((32+z, 322+xy, 322+xy), 'predict_net', 0, [4,4,4], [1,20,20,20,1], 0.4, double_suppression_size=10, add_noise=True)
