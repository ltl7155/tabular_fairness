from tensorflow import keras
import os
import joblib
import numpy as np
from explain import  get_relevance, get_critical_neurons
import tensorflow as tf
from tensorflow import set_random_seed
from scalelayer import  ScaleLayer
from numpy.random import seed

import argparse

seed(1)
set_random_seed(2)

for i in range(5):
    tmp_data = data.copy()
    tmp_data[:, 6] = i
    new_data.append(tmp_data)

def loss_fun2(y_true, y_pred):
    # do whatever you want
    return y_pred

def construct_model(neurons, top_layer, name, min, max, need_weights=True):
    in_shape = X_train.shape[1:]
    input = keras.Input(shape=in_shape)
    # layer1 = keras.layers.Dense(30, activation="relu", name="layer1")
    # d1 = ScaleLayer(30, min, max)
    # layer2 = keras.layers.Dense(20, activation="relu", name="layer2")
    # d2 = ScaleLayer(20, min, max)
    # layer3 = keras.layers.Dense(15, activation="relu", name="layer3")
    # d3 = ScaleLayer(15, min, max)
    # layer4 = keras.layers.Dense(15, activation="relu", name="layer4")
    # d4 = ScaleLayer(15, min, max)
    # layer5 = keras.layers.Dense(10, activation="relu", name="layer5")
    # d5 = ScaleLayer(10, min, max)
    # layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    layer1 = keras.layers.Dense(30, name="layer1")
    d1 = ScaleLayer(30, min, max)
    layer2 = keras.layers.Dense(20, name="layer2")
    d2 = ScaleLayer(20, min, max)
    layer3 = keras.layers.Dense(15, name="layer3")
    d3 = ScaleLayer(15, min, max)
    layer4 = keras.layers.Dense(15, name="layer4")
    d4 = ScaleLayer(15, min, max)
    layer5 = keras.layers.Dense(10,name="layer5")
    d5 = ScaleLayer(10, min, max)
    layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")

    layer_lst = [layer1, layer2, layer3, layer4, layer5]
    ds = [d1, d2, d3, d4, d5]
    for layer in layer_lst[0: top_layer]:
        layer.trainable = False

    x = input
    for i, l in enumerate(layer_lst):
        x = l(x)
        if i < top_layer:
            x = ds[i](x)
    x = layer6(x)

    if not need_weights:
        return keras.Model(input, x)

    w = 0.
    for i, re in enumerate(neurons):
        neg = re[0]
        pos = re[1]
        d = ds[i]
        for m in neg:
            w = tf.math.add(w, d.weights[0][0][m])
        for n in pos:
            w = tf.math.subtract(w, d.weights[0][0][n])
    new_w = tf.identity(tf.reshape(w, [1, 1]), name=name)

    model = keras.Model(input, [x, new_w])

    return model

def get_acc(newdata_res):
    for i in range(4):
        tmp_acc = (newdata_res[i] == newdata_res[i + 1]) * 1
        if i == 0:
            acc = tmp_acc
        else:
            acc += tmp_acc
    return np.sum(np.where(acc == 4, True, False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--path', default='models/adult_model.h5', help='model_path')
    parser.add_argument('--target_model_path', default='models/adult_EIDIG_INF_retrained_model.h5', help='model_path')
    parser.add_argument('--attr', default='g', help='protected attributes')
    args = parser.parse_args()
    attr = args.attr
    model_path = "models/finetuned_models_protected_attributes/"
    path_ls = os.listdir(model_path)
    path_dict = {}
    path_dict['r'] = [model_path+p for p in path_ls if "r_adult" in p].sort()
    path_dict['g'] = [model_path+p for p in path_ls if "g_adult" in p].sort()
    path_dict['a'] = [model_path+p for p in path_ls if "a_adult" in p].sort()

    X_train, X_val, y_train, y_val, y_sex_train, y_sex_val, constraint = joblib.load('data/adult.data')
    if attr != "":
        data = np.load("data/C-r_ids_EIDIG_INF.npy")
    new_data = []
    target_model_path = args.target_model_path

    income_path = 'models/adult_model.h5'
    income_train_scores = get_relevance(income_path, X_train, save_path=os.path.join('scores', os.path.basename(income_path) + ".score"))
    percent = 0.3
    weight_threshold = 0.2
    income_critical = get_critical_neurons(income_train_scores, percent)
    saved = False

    finals = []
    for top_n in [4]:
        attrs = attr.split("&")
        protected_critical_ls = []
        for a in attrs:
            path = path_dict[a][top_n-1]
            train_scores = get_relevance(path, X_train, save_path=os.path.join('scores', os.path.basename(path) + ".score"))
            protected_critical = get_critical_neurons(train_scores, percent)
            protected_critical_ls.append(protected_critical)

        layer_num = len(income_critical)
        total_num = len(X_train)
        results = []
        def my_filter(layer_critical):
            i_unique, i_counts = np.unique(layer_critical, return_counts=True)
            i_rates = i_counts / total_num
            i_sort = np.where(i_rates > weight_threshold)[0]  # np.argsort(i_counts*-1)
            i_critical = i_unique[i_sort]

            return i_critical

        for i in range(layer_num):
            income_layer_critical = income_critical[i].flatten()
            i_critical = my_filter(income_layer_critical)
            current_penalty = None
            current_awarded = None
            filtered_criticals = []
            for j, a in enumerate(attrs):
                protected_layer_critical = protected_critical_ls[j][i].flatten()
                p_critical = my_filter(protected_layer_critical)
                filtered_criticals.append(p_critical)
                penalty = np.setdiff1d(p_critical, i_critical)
                awarded = np.setdiff1d(i_critical, p_critical)
                if current_penalty == None:
                    current_penalty = penalty
                else:
                    current_penalty = np.union1d(current_penalty, penalty)
                if current_awarded == None:
                    current_awarded = awarded
                else:
                    current_awarded = np.setdiff1d(current_awarded, awarded)
            results.append((penalty, awarded))

        results = results[1:(top_n + 1)]
        # paras = [(-1, 1), (-0.8, 1), (-0.5, 1), (-0.2, 1), (0, 1), (-1, 1.5), (-0.8, 1.5), (-0.4, 1.5), (0, 1.5), (-1, 2), (-0.8, 2), (-0.4, 2), (0,2)]
        # paras = [(0.2, 1), (0.5, 1), (0.7,1), (0.9,1)]
        paras = [(-1, 1)]
        para_res = dict()
        for k, ps in enumerate(paras):
            name = 'my_name_' + str(top_n) + '_' + str(k)
            new_model = construct_model(results, top_n, name, ps[0], ps[1])
            new_model.load_weights(income_path, by_name=True)
            tf_name = 'tf_op_layer_' + name
            losses = {'layer6': 'binary_crossentropy', tf_name: loss_fun2}
            losses_weights = {'layer6': 1.0, tf_name: 1}

            new_model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics={'layer6': "accuracy"})
            history = new_model.fit(x=X_train, y={'layer6': y_train, tf_name: y_train}, epochs=10,
                                    validation_data=(X_val, {'layer6': y_val, tf_name: y_val}))

            re, _ = new_model.predict(X_val)
            data_maskmodel = (re > 0.5).astype(int).flatten()

            val_acc = np.sum(data_maskmodel == y_val) / len(y_val)

            # data_re, _ = new_model.predict(data)
            # data_re = (data_re > 0.5).astype(int).flatten()
            newdata_res = []
            for i in range(5):
                newdata_re, _ = new_model.predict(new_data[i])
                newdata_re = (newdata_re > 0.5).astype(int).flatten()
                newdata_res.append(newdata_re)

            repair_acc = get_acc(newdata_res) / len(data)
            finals.append((val_acc, repair_acc))
            # print(results)
            para_res[ps] = (val_acc, repair_acc)

            if saved:
                model_name = 'models/race_gated_'+str(top_n)+'_'+str(percent)+'_'+str(weight_threshold)+'.h5'
                saved_model = construct_model(results, top_n, name, need_weights=False)
                saved_model.set_weights(new_model.get_weights())
                saved_model.trainable = True
                tf.keras.models.save_model(saved_model, model_name)
        for k in para_res.keys():
            print(k, para_res[k])
            # weights = new_model.get_weights()


#
# income_train_scores = get_relevance(income_path, data, save_path=os.path.basename(income_path) + "_data.score")
# race_train_scores = get_relevance(race_path, data, save_path=os.path.basename(race_path) + "_data.score")

#
# income_train_scores = get_relevance(income_path, new_data, save_path=os.path.basename(income_path) + "_ndata.score")
# race_train_scores = get_relevance(race_path, new_data, save_path=os.path.basename(race_path) + "_ndata.score")

# print(finals)


augmented_model = keras.models.load_model(path1)
aug_val = (augmented_model.predict(X_val) > 0.5).astype(int).flatten()
aug_data = (augmented_model.predict(data) > 0.5).astype(int).flatten()
newdata_res = []
for i in range(5):
    newdata_re = augmented_model.predict(new_data[i])
    newdata_re = (newdata_re > 0.5).astype(int).flatten()
    newdata_res.append(newdata_re)
# aug_newdata = (augmented_model.predict(new_data) > 0.5).astype(int).flatten()

print('Aug', np.sum(aug_val == y_val)/len(y_val))
print('Aug', get_acc(newdata_res)/len(data))

