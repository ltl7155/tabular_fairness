"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""
from tensorflow import keras
import numpy as np
import sys, os
import tensorflow as tf
from tensorflow.keras import backend as K

sys.path.append(".")

from evaluation import experiments
from evaluation import generation_utilities
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing


"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    # compute the percentage of individual discriminatory instances with 95% confidence

    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = generation_utilities.purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)
    return avg, interval

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = K.variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, min, max))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import argparse

def get_acc(y_true, y_pred):
    print('The acc is', np.sum(y_true == y_pred) / len(y_pred))
    return np.sum(y_true == y_pred) / len(y_pred)

def get_models():
    # saved_model_path = "models/gated_models/"
    # path_adult = ["adult_a_gated_4_0.3_0.2_p-0.5_p1.0.h5", "adult_r_gated_4_0.3_0.2_p-0.8_p1.0.h5", "adult_g_gated_4_0.3_0.2_p0.0_p1.5.h5",
    #               "adult_a&r_gated_4_0.3_0.2_p-1.0_p2.0.h5", "adult_a&g_gated_4_0.3_0.2_p-1.0_p1.5.h5", "adult_r&g_gated_4_0.3_0.2_p-1.0_p2.0.h5"]
    # path_german = ["german_g_gated_4_0.3_0.2_p-0.8_p1.0.h5", "german_a_gated_4_0.3_0.2_p-0.2_p1.0.h5", "german_g&a_gated_4_0.3_0.2_p-0.8_p2.0.h5"]
    # path_bank = ["bank_a_gated_4_0.3_0.2_p-0.8_p1.0.h5"]

    saved_model_path = "models/gated_models/"
    path_adult = ["adult_a&g_gated_4_0.3_0.2_p-0.3_p0.3.h5"]
    path_german = ["german_g_gated_4_0.3_0.2_p0.2_p0.9.h5", "german_a_gated_4_0.3_0.2_p-1.0_p0.2.h5"]
    path_bank = ["bank_a_gated_4_0.3_0.2_p-0.8_p1.0.h5"]

    path_adult = [saved_model_path + p for p in path_adult]
    path_german = [saved_model_path + p for p in path_german]
    path_bank = [saved_model_path + p for p in path_bank]
    path_dict = [path_adult, path_german, path_bank]
    return path_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--i', type=int)
    args = parser.parse_args()

    model_lists = get_models()
    ROUND = 1
    results = {}
    dataset_index = args.i
    models = model_lists[dataset_index]
    dataset_modules = [pre_census_income, pre_german_credit, pre_bank_marketing]
    dataset_module = dataset_modules[dataset_index]
    # attr_lists = [[('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0, 6]), ('C-a&g', [0, 7]), ('C-r&g', [6, 7])],
    #               [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6, 9])],
    #               [('B-a:', [0])]]
    attr_lists = [[('C-a&g', [0, 7])],
                 [('G-g', [6]), ('G-a', [9])],
                 [('B-a:', [0])]]
    attr_list = attr_lists[dataset_index]
    id_seeds = experiments.generate_seeds(dataset_module.X_train, num_seeds=1000)

    for i in range(len(models)):
        m = models[i]
        print("For model" + m)
        results[m] = {}
        model = keras.models.load_model(m, custom_objects={'ScaleLayer': ScaleLayer})

        print("______Calculating acc of model " + m)
        acc = get_acc(dataset_module.y_test, (model.predict(dataset_module.X_test) > 0.5).astype(int).flatten())
        results[m]['acc'] = round(acc, 3)
        print("______Calculating global_comparison of model " + m)
        results[m]['global'] = {}
        benchmark, protected_attribs = attr_list[i]
        b = os.path.basename(m) +'_'+ benchmark
        print('\n Global_comparison', b, ':\n')
        num_ids, num_iter, time_cost = experiments.my_global_comparison(ROUND, benchmark, dataset_module.X_train, id_seeds,
                                                                     protected_attribs,
                                                                     dataset_module.constraint, model,
                                                                     [0.5])
        results[m]['global'][benchmark] = num_ids
        print("______Calculating purely_random of model " + m)
        results[m]['random'] = {}
        benchmark, protected_attribs = attr_list[i]
        avg, interval = ids_percentage(10, 100, len(dataset_module.X[0]), protected_attribs,
                       dataset_module.constraint, model)
        results[m]['random'][benchmark] = str(round(avg, 3)) + "+-" + str(round(interval, 3))

    print(results)



