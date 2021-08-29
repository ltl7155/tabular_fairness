from tensorflow import keras
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed
from tensorflow.keras.utils import to_categorical
from preprocessing import pre_census_income
from scalelayer import  ScaleLayer

seed(1)
set_random_seed(2)
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

X_train, X_val, y_train, y_val, constraint = pre_census_income.X_train, \
    pre_census_income.X_val, pre_census_income.y_train, pre_census_income.y_val, pre_census_income.constraint

# a = X_train[:, 0]
# r = X_train[:, 6]
# g = X_train[:, 7]
# print(np.unique(a, return_counts=True))
# print(np.unique(r, return_counts=True))
# print(np.unique(g, return_counts=True))

# y_train_race = to_categorical(X_train[:, 6], num_classes=5)
# y_val_race = to_categorical(X_val[:, 6], num_classes=5)
# print(y_train_income.shape, y_train_race.shape)
# print(np.unique(y_train_race, return_counts=True))
#
# data = np.load("data/C-g_ids_EIDIG_INF_1_5db56c7ebc46082e507dc3145ff8fcd6.npy")


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
#     parser.add_argument('--path', default='models/retrained_models_EIDIG/adult_EIDIG_INF_retrained_model.h5', help='model_path')
    parser.add_argument('--attr', default='a', help='protected attributes')
    args = parser.parse_args()

    pos_map = { 'a': 0,
            'r': 6,
            'g': 7,
            }
    category_map = {'a': 4,
               'r': 5,
               'g': 1,
               }
    frozen_layers = [1, 2, 3, 4, 5]
    
    models_map = {
        'a': "models/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15.h5",
        'r': "models/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1.h5",
        'g': "models/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8.h5",
#         'a&r': "models/adult_a&r_gated_4_0.3_0.2_p-0.4_p0.5.h5",
#         'a&g': "models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.2.h5",
#         'r&g': "models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8.h5",
    }

    for frozen_layer in frozen_layers:
        model_path = models_map[args.attr]
        model = keras.models.load_model(model_path, custom_objects={'ScaleLayer': ScaleLayer})
        
        inner_model = Model(model.input, model.get_layer(layer_name).output)                                 
        inner_output = inner_model.predict(pre_census_income.X_train)
        
        
        
        # attrs = args.a.split('&')
        attr = args.attr
        losses = {}
        losses_weights = {}
        metrics = {}
        y_train_labels = {}
        y_val_labels = {}
        last_layer_name = 'layer_' + attr
        if attr == "g":
            losses[last_layer_name] = 'binary_crossentropy'
        else:
            losses[last_layer_name] = 'categorical_crossentropy'
        losses_weights[last_layer_name] = 1.0
        metrics[last_layer_name] = "accuracy"
        if attr == "g":
            y_train_labels[last_layer_name] = X_train[:, pos_map[attr]]
            y_val_labels[last_layer_name] = X_val[:, pos_map[attr]]
        elif attr == "a":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]]-1,
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]]-1,
                                                               num_classes=category_map[attr])
        elif attr == "r":
            y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]],
                                                             num_classes=category_map[attr])
            y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]],
                                                               num_classes=category_map[attr])

        # nadam = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics=metrics)

        history = model.fit(x=X_train, y=y_train_labels, epochs=30,
                            validation_data=(X_val, y_val_labels))
        # save model.
        model_name = 'models/finetuned_models_protected_attributes3/adult/' + args.attr + '_adult_model_' + str(frozen_layer) + "_" + str(round(history.history["val_acc"][-1], 3)) + '.h5'
        keras.models.save_model(model, model_name)
