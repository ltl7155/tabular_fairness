import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np 

from tensorflow import keras
import os
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
np.random.seed(42)

import pre_mnist_01
import mnist_ori_09


def construct_model(frozen_layers, attr_info,archs=[32,16]):
    input = keras.Input(shape=(28,28))
    flatten_layer =  keras.layers.Flatten(input_shape=(28, 28))

    layer_list = [
        keras.layers.Dense(xi, activation="relu", name=f"layer_{ii}")
        for ii,xi in enumerate(archs) 
        ]

    attr_name = attr_info["name"]
    attr= attr_name
    attr_class_num = attr_info["class_num"]
    
    last_layer_maintask = keras.layers.Dense(attr_class_num, activation="sigmoid", name='output')
    
    if attr_class_num==1 :
        last_layer = keras.layers.Dense(attr_class_num, activation="sigmoid", name='layer_' + attr)
    else:
        last_layer = keras.layers.Dense(attr_class_num, activation="softmax", name='layer_' + attr)

    
    for layer in layer_list[0: frozen_layers]:
        layer.trainable = False
    x = input
    x = flatten_layer(x)
    for i, l in enumerate(layer_list):
        x = l(x)
    # y_income = layer6(x)
    y = last_layer(x)
    y_main = last_layer_maintask(x)
    
    model = keras.Sequential([
        input,
        flatten_layer,
        *layer_list,
        last_layer,
        ])
    model_maintask = keras.Sequential([
        input,
        flatten_layer,
        *layer_list,
        last_layer_maintask,
        ])
    
    return model, model_maintask#keras.Model(input, [y_main,y])

def evaluate_model(model,x_train,y_train):
    y_hat_logist=  model.predict(x_train)
    y_hat = (y_hat_logist>0.5).astype(np.int32).flatten()
    acc = np.sum(y_hat==y_train) /len(y_train)
    return acc 
def evaluate_model_ensemble(model_header,model_last,x_train,y_train):
    '''
        x=model1.layer1(x)
        x=model1.layer2(x)
        x=model1.layer3(x)
        x=model1.layer4(x)
        #y=model1.last(x)
        y=model2.last(x)
    '''
    x_in = model_header.layers[0].input 
    tmp_int = model_header.layers[-2].output
    final_y = model_last.layers[-1](tmp_int) 
    x_model = keras.Model(x_in,final_y)
    
    y_hat_logist=  x_model.predict(x_train)
    y_hat = (y_hat_logist>0.5).astype(np.int32).flatten()
    acc = np.sum(y_hat==y_train) /len(y_train)
    return acc 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument("-r","--rate",type=float,default=0.2)
    parser.add_argument("-l","--labels",choices=list(map(str, range(0,10))),default=[0,9],nargs="+")
    parser.add_argument("-e","--epochs",default=5)
    parser.add_argument("-n","--net_layers",default="32,16",type=str,help="arch by comma")

    parser.add_argument('--path', default='./models/original/mnist01_model_c14fbf8c633c593f029630ad41bd2c7b.h5', help='model_path')
    args = parser.parse_args()

    attr_info ={"name":"background","class_num":1}
    net_arch_args= args.net_layers
    net_arch_args= [int(x) for x in net_arch_args.split(",")] 
    
    frozen_layers = range(1,len(net_arch_args)+1)

    attr = attr_info["name"]
    attr_class_num = attr_info["class_num"]
    
    ##################
    ####dataset 
    ##################
    flip_rate=args.rate
    # label_choices=[0,9]
    label_choices=args.labels
    label_choices= [int(x) for x in label_choices]
    ##load 
    dataset_dict = pre_mnist_01.func_call(flip_rate=flip_rate,label_choices= label_choices)
    X_train = dataset_dict["X_train"]
    y_train = dataset_dict["y_train"]
    X_val = dataset_dict["x_val"]
    y_val = dataset_dict["y_val"]
    y_train_protected = dataset_dict["y_train_protected"]
    y_val_protected = dataset_dict["y_val_protected"]
    
        
    for frozen_layer in frozen_layers:
        model , model_multitask  = construct_model(frozen_layer, attr_info=attr_info,archs=net_arch_args)
        model.load_weights(args.path, by_name=True)
        
        from sklearn.utils import compute_class_weight
        classWeight = compute_class_weight('balanced', [0,1], y_train) 
        classWeight = dict(enumerate(classWeight))
        print (classWeight)
        
        model_multitask.summary()
        model_multitask.load_weights(args.path, by_name=True)
        
        acc_main1 = evaluate_model_ensemble(model_header=model,model_last=model_multitask,
                                 x_train=X_train, 
                                 y_train=y_train,
                                )
        print (acc_main1,"acc.main.1")
        acc_main = evaluate_model(model=model_multitask,
                                 x_train=X_train, 
                                 y_train=y_train,
                                 )
        print ("main","acc",acc_main  )

        mnist_ori_09.display_images(
            images=X_train[:32],
            titles=["label_{}".format(x) for x in y_train[:32]],
            save_filename=os.path.join("./images","{}.jpg".format("main")),
            )
        mnist_ori_09.display_images(
            images=X_train[:32],
            titles=["label_{}".format(x) for x in y_train_protected[:32]],
            save_filename=os.path.join("./images","{}.jpg".format("bg")),
            )
 
        
        # nadam = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss="binary_crossentropy", 
                       optimizer="adam", metrics=["accuracy"])

        history = model.fit(x=X_train, y=y_train_protected, epochs=10,
                            validation_data=(X_val, y_val_protected))

        
        
        # save model.json 
        model_name = 'models/finetuned_models_protected_attributes2/mnist01/' + attr + '_mnist01_model_' + str(frozen_layer) + "_" + str(round(history.history["val_accuracy"][-1], 3)) + '.h5'
        keras.models.save_model(model, model_name)
        
        # save model.
        args_dict= vars(args)
        args_dict.update({"attr":attr,"frozen_layer":frozen_layer,"history":history.history})
        meta_json_path = model_name.replace(".h5",".json")
        import json 
        with open(meta_json_path,"w") as f :
            json.dump(obj=args_dict,fp=f,indent=2)





