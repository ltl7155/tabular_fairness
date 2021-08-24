"""
This python file provides experimental functions backing up the claims involving efficiency and effectiveness in our paper.
"""


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from . import generation_utilities
from . import ADF
from . import EIDIG


# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# make outputs stable across runs for validation
# alternatively remove them when dealing with real-world issues
np.random.seed(42)
tf.random.set_seed(42)


def my_global_comparison(num_experiment_round, benchmark, X, seeds, protected_attribs, constraint, model, decay_list,
                      num_seeds=1000, c_num=4, max_iter=10, s_g=1.0):
    # compare the global phase given the same set of seeds

    num_ids = np.array([0] * (len(decay_list) + 1))
    num_iter = np.array([0] * (len(decay_list) + 1))
    time_cost = np.array([0] * (len(decay_list) + 1))

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        num_dis = 0
        for seed in seeds:
            similar_seed = generation_utilities.similar_set(seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(seed, similar_seed, model):
                num_dis += 1
        print('Given', len(seeds), '(no more than 600 for german credit) seeds,', num_dis,
              'of which are individual discriminatory instances.')

        t1 = time.time()
        ids_ADF, _, total_iter_ADF = ADF.global_generation(X, seeds, num_attribs, protected_attribs, constraint, model,
                                                           max_iter, s_g)
        t2 = time.time()
        num_ids_ADF = len(ids_ADF)
        print('ADF:', 'In', total_iter_ADF, 'search iterations,', num_ids_ADF,
              'non-duplicate individual discriminatory instances are generated. Time cost:', t2 - t1, 's.')
        num_ids[0] += num_ids_ADF
        num_iter[0] += total_iter_ADF
        time_cost[0] += t2 - t1

        for index, decay in enumerate(decay_list):
            # print('Decay factor set to {}:'.format(decay))
            t1 = time.time()
            ids_EIDIG, _, total_iter_EIDIG = EIDIG.global_generation(X, seeds, num_attribs, protected_attribs,
                                                                     constraint, model, decay, max_iter, s_g)
            t2 = time.time()
            num_ids_EIDIG = len(ids_EIDIG)
            print('EIDIG:', 'In', total_iter_EIDIG, 'search iterations,', num_ids_EIDIG,
                  'non-duplicate individual discriminatory instances are generated. Time cost:', t2 - t1, 's.')
            num_ids[index + 1] += num_ids_EIDIG
            num_iter[index + 1] += total_iter_EIDIG
            time_cost[index + 1] += t2 - t1

        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    avg_iter = num_iter / num_experiment_round / num_seeds
    print('Results of global phase comparsion on', benchmark, 'given {} seeds'.format(num_seeds), ',averaged on',
          num_experiment_round, 'rounds:')
    print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0],
          'per second, and the number of iterations on a singe seed is', avg_iter[0], '.')
    for index, decay in enumerate(decay_list):
        print('Decay factor set to {}:'.format(decay))
        print('EIDIG:', avg_num_ids[index + 1], 'individual discriminatory instances are generated at a speed of',
              avg_speed[index + 1], 'per second, and the number of iterations on a singe seed is', avg_iter[index + 1],
              '.')

    return num_ids, num_iter, time_cost



def generate_seeds(X, c_num=4, num_seeds = 100, fashion='Distribution'):
    num_attribs = len(X[0])
    clustered_data = generation_utilities.clustering(X, c_num)
    id_seeds = np.empty(shape=(0, num_attribs))
    for i in range(100000000):
        x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
        id_seeds = np.append(id_seeds, [x_seed], axis=0)
        if len(id_seeds) >= num_seeds:
            break
    # for i in range(num_seeds):
    #     x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion='Distribution')
    #     seeds = np.append(seeds, [x_seed], axis=0)
    return id_seeds


