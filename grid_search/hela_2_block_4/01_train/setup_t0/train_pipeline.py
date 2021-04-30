from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
from lsd.gp import AddLocalShapeDescriptor
import os
import math
import json
import tensorflow as tf
import numpy as np
from micron import read_train_config


def train_until(max_iteration,
                training_container,
                raw_dset,
                gt_dset):

    """
    max_iteration [int]: Number of training iterations

    data_dir [string]: Training data base directory

    samples [list of strings]: hdf5 files holding the training data. Each 
                             file is expected to have a dataset called
                             *raw* holding the raw image data and 
                             a dataset called *tracing* holding the microtubule
                             tracings.
    """

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    soft_mask = ArrayKey('SOFT_MASK')
    pred_maxima = ArrayKey('PRED_MAXIMA')
    pred_reduced_maxima = ArrayKey('PRED_REDUCED_MAXIMA')

    voxel_size = Coordinate(config['voxel_size'])
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(soft_mask, input_size)
    
    data_sources = tuple(
        Hdf5Source(
            container,
            datasets = {
                soft_mask: raw_dset
            },
            array_specs = {
                soft_mask: ArraySpec(interpolatable=True)
            }
        ) +
        Normalize(soft_mask) +
        Pad(soft_mask, None) +
        RandomLocation()
        for container in training_container
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['soft_mask']: soft_mask,
            },
            outputs={
                config['soft_mask']: soft_mask,
                config['derivatives']: derivatives,
                config['loss_weights_lsds']: loss_weights_lsds,
                config['gt_maxima']: gt_maxima,
                config['gt_reduced_maxima']: gt_reduced_maxima,
                config['pred_maxima']: pred_maxima,
                config['pred_reduced_maxima']: pred_reduced_maxima
            },
            gradients={},
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'raw',
                tracing: 'tracing',
                gt_lsds: 'gt_lsds',
                soft_mask: 'soft_mask',
                derivatives: 'derivatives',
                loss_weights_lsds: 'loss_weights_lsds',
                gt_maxima: 'gt_maxima',
                gt_reduced_maxima: 'gt_reduced_maxima',
                pred_maxima: 'pred_maxima',
                pred_reduced_maxima: 'pred_reduced_maxima'
            },
            dataset_dtypes={
                tracing: np.uint64
            },
            every=1000,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    train_config = read_train_config("./train_config.ini")
    train_config["max_iteration"] = iteration

    train_until(**train_config)
