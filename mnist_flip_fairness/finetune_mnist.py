import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np 

from tensorflow import keras
import os
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical

#tf.random.set_seed(42)
if str(tf.__version__).startswith("2."):
    tf.random.set_seed(42)
else:
    tf.random.set_random_seed(42)

np.random.seed(42)

import pre_mnist_01
# import mnist_ori_09

dataset_mnist =pre_mnist_01.func_call(flip_rate=0.2,label_choices=[0,9])
X_train = dataset_mnist["X_train"]
X_val = dataset_mnist["x_val"]

def construct_model(frozen_layers, attr,args_arch):
    in_shape = X_train.shape[1:]
    
    input =keras.Input(shape=in_shape,name="input")
    
    args_arch= [int(x) for x in args_arch.split(",")] 
    archs = [tf.keras.layers.Dense(x, activation="relu",name=f"layer_{ii}")
             for ii,x in enumerate(args_arch) 
             ]

    # layer1 = keras.layers.Dense(64, activation="relu", name="layer_1")
    # layer2 = keras.layers.Dense(32, activation="relu", name="layer_2")
    # layer3 = keras.layers.Dense(32, activation="relu", name="layer_3")
    # layer4 = keras.layers.Dense(16, activation="relu", name="layer_4")
    # layer6 = keras.layers.Dense(1, activation="sigmoid", name="layer6")
    c = category_map[attr]
    if attr == 'background':
        last_layer = keras.layers.Dense(c, activation="sigmoid", name='layer_' + attr)
    # else:
    #     last_layer = keras.layers.Dense(c, activation="softmax", name='layer_' + attr)
    # layer_lst = [layer1, layer2, layer3, layer4]
    layer_lst = archs
    for layer in layer_lst[0: frozen_layers]:
        layer.trainable = False
    x = input
    for i, l in enumerate(layer_lst):
        x = l(x)
    # y_income = layer6(x)
    y = last_layer(x)
    model = keras.Sequential([input, *archs, last_layer])
    # return keras.Model(input, y_race)
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--path', default='models/original/mnist01_model_onlyweight_988cf647eee71fb30938c3c2cb0df718.h5', help='model_path')
    parser.add_argument('--attr', default='background', help='protected attributes')
    parser.add_argument("-n","--net_layers",default="64,32,32,16,10",type=str,help="arch by comma")
    args = parser.parse_args()

    assert args.attr in ["background"]
    attr_info ={"name":"background","class_num":1}
    category_map= {attr_info["name"]:attr_info["class_num"]}
    # net_arch_args= args.net_layers
    # net_arch_args= [int(x) for x in net_arch_args.split(",")] 
    
    frozen_layers = [1, 2, 3, 4, 5]

    for frozen_layer,epochs in zip(frozen_layers, [10,10,20,30,50]):
        model = construct_model(frozen_layer, args.attr,args_arch=args.net_layers)
        model.load_weights(args.path, by_name=True)# if ".tf" in args.path else False )

        attr = args.attr
        losses = {}
        losses_weights = {}
        metrics = {}
        y_train_labels = {}
        y_val_labels = {}
        last_layer_name = 'layer_' + attr
        if attr == "background":
            losses[last_layer_name] = 'binary_crossentropy'
        # else:
        #     losses[last_layer_name] = 'categorical_crossentropy'
        losses_weights[last_layer_name] = 1.0
        metrics[last_layer_name] = "accuracy"
        if attr == "background":
            y_train_labels[last_layer_name] =dataset_mnist["y_train_protected"]# X_train[:, pos_map[attr]]
            y_val_labels[last_layer_name] = dataset_mnist["y_val_protected"]#X_val[:, pos_map[attr]]
        # elif attr == "a":
        #     y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]]-1,
        #                                                      num_classes=category_map[attr])
        #     y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]]-1,
        #                                                        num_classes=category_map[attr])
        # elif attr == "r":
        #     y_train_labels[last_layer_name] = to_categorical(X_train[:, pos_map[attr]],
        #                                                      num_classes=category_map[attr])
        #     y_val_labels[last_layer_name] = to_categorical(X_val[:, pos_map[attr]],
                                                               # num_classes=category_map[attr])

        # nadam = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=losses, loss_weights=losses_weights, optimizer="nadam", metrics=metrics)
        history = model.fit(x=X_train, y=y_train_labels, epochs=epochs,
                            validation_data=(X_val, y_val_labels))
        val_accuracy = history.history["val_acc"] if "val_acc" in history.history else history.history["val_accuracy"]
        # # save model.
        model_name = 'models/finetuned_models_protected_attributes2/mnist01/' + args.attr + '_mnist01_model_' + str(frozen_layer) + "_" + str(round(val_accuracy[-1], 3)) + '.h5'
        keras.models.save_model(model, model_name)



