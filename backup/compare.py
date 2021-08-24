"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""

import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras
import numpy as np
import os


"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""


import tensorflow as tf
class  ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = tf.Variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, -1, 1.5))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-i', help='model_path')
    args = parser.parse_args()


    models = ['models/adult_model.h5',
            'models/adult_EIDIG_5_retrained_model.h5',
              'models/adult_EIDIG_INF_retrained_model.h5',
              'models/repaired_models/race_gated_layer4_per0.3_thresh0.2_va0.821_ra0.993.h5',
              ]


    ROUND = 1
    update_interval_list = np.append(np.arange(1, 11, 1), 10000)

    id_seeds = experiments.generate_seeds(pre_census_income.X_train, num_seeds=1000)

    for benchmark, protected_attribs in [('C-r', [6])]:

        for m in  models:
            b = os.path.basename(m) +'_'+ benchmark
            print('\n', b, ':\n')
            adult_model = keras.models.load_model(m, custom_objects={'ScaleLayer': ScaleLayer})
            num_ids, num_iter, time_cost = experiments.my_global_comparison(ROUND, benchmark, pre_census_income.X_train, id_seeds,
                                                                         protected_attribs,
                                                                         pre_census_income.constraint, adult_model,
                                                                         [0.5])

            # experiments.my_local_comparison(ROUND, benchmark, pre_census_income.X_train, id_seeds, protected_attribs,
            #                              pre_census_income.constraint, adult_model, update_interval_list)

        # experiments.comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs,
        #                        pre_census_income.constraint, adult_model, g_num=100, l_num=500)



