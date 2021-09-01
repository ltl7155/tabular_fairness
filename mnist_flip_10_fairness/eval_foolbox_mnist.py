#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np 
import json 

from datetime import datetime 


import foolbox_attack_mnist as foolbox_util 

def create_model(weight_path,net_layers=[64,32,16,16,10]):

    layer_lst = [
        tf.keras.layers.Dense(xi, activation="relu", name=f"layer_{ii}")
                for ii,xi in enumerate(net_layers) 
                        ]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        *layer_lst,
        tf.keras.layers.Dense(1, activation="sigmoid",name="output"),
    ])
    model.load_weights(weight_path,by_name=True)
    #"./models/original/mnist01_model_onlyweight_988cf647eee71fb30938c3c2cb0df718.h5",by_name=True)
    return model

def load_from_adv_with_epslist(npz_path):
    datalist = np.load(npz_path,allow_pickle=True)["data"]
    adv_list = [x["adv_images"] for x in datalist ]
    return adv_list



def main(args) -> None:



    save_dir =args. save_dir
    name_prefix=os.path.splitext( os.path.basename(args.weight_path))[0]

    print ("will save as",name_prefix)



    # instantiate a model (could also be a TensorFlow or JAX model)
    model =create_model(weight_path=args.weight_path,net_layers  =args.net_layers )

    ####
    ##load data
    ####
    (_,_),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    y_test_index09 =np.logical_or( y_test==0,y_test==9) 
    x_test = (x_test[y_test_index09].astype(np.float))/255.
    y_test = y_test[y_test_index09]
    y_test[y_test==9]=1
    
    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels =tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.int64)
    
    ####
    ##check load weight
    ####
    #check the model's loading  is correct or not ?
    naive_acc_info = foolbox_util.get_acc(model=model, labels= labels ,images=images) 
    print ("test before being attacked ")
    print("\t clean(naive) accuracy:  {naive_acc:.4f} %".format( naive_acc=naive_acc_info["acc"]*100 ))


    ####
    ##attack with eps
    ####
    print ("load from saved npz,",args.adv_path)
    cliped_adv_images = load_from_adv_with_epslist(args.adv_path) 
    assert type(cliped_adv_images)==list 

    #assert_func  =lambda x: assert(type(x)==np.ndarray) 

    np_to_list = lambda x:x.tolist() if "numpy" in str(type(x)) else x
    tf_to_np = lambda x: x.numpy() if tf.is_tensor(x) else x

    save_data_list=[]
    save_info_list=[]
    from tqdm import tqdm 
    for ids,adv_images  in tqdm(enumerate( cliped_adv_images),total=len(cliped_adv_images) ):
        inconsis_info =foolbox_util. get_inconsistent_rate(model=model,images=adv_images,labels=labels)

        pred_y_hat =tf_to_np( inconsis_info["y_hat"] )
        pred_y_hat_n =tf_to_np( inconsis_info["y_hat_n"] )
        adv_images = tf_to_np( adv_images )
        assert type(adv_images) ==np.ndarray , type(adv_images)
        assert type(pred_y_hat) ==np.ndarray , type(pred_y_hat)
        assert type(pred_y_hat_n) ==np.ndarray , type(pred_y_hat_n)

        #data_info ={"adv_images":adv_images,"pred_y_hat":pred_y_hat,"pred_y_hat_n":pred_y_hat_n,"eps":eps}
        #save_data_list.append(data_info)


        inconsis_info_debug = {k:np_to_list(v) for k,v in inconsis_info.items() if k in ["acc","acc_fairness","inconsis_c","total_c","inconsistent_rate"]}
        save_info_list.append(inconsis_info_debug)


    #save_data_path = os.path.join(save_dir,"{name_prefix}_adv.npz".format(name_prefix=name_prefix))
    #np.savez(save_data_path ,data=save_data_list)

    logid = datetime.now()
    logid = logid.strftime("%m-%d-%Y_%H-%M-%S")

    save_meta_path  = os.path.join(save_dir,"{name_prefix}__{logid}__meta.json".format(name_prefix=name_prefix,logid=logid))
    print ("info save in ",save_meta_path)
    with open( save_meta_path ,"w") as f :
        json.dump(fp=f,obj=save_info_list,indent=2)


    #{"acc":acc,"acc_fairness":acc_fairness,"inconsis_c":inconsistent_c,"total_c":total_c,"inconsistent_rate":rate,"y_hat":y_hat,"y_hat_n":y_hat_n}

    #predicted_labels_eps_like =reverse_predict_label_from_foolboxsuccess(success_attack=success_foolbox,labels=labels)
    ####
    ##run fairness metric
    ####
    #fairness_info = unfair_metric(clean_predict=clean_predict,pred_labels_eps_like=predicted_labels_eps_like,eps_list=eps_list)
    #print ("fairness_info","....",fairness_info)
    


if __name__ == "__main__":
    assert str(tf.__version__).startswith("2."), ("tf.version>2.0", tf.__version__)
    import eagerpy as ep
    from foolbox import TensorFlowModel, accuracy, samples, Model
    from foolbox.attacks import LinfPGD

    import argparse
    parser = argparse.ArgumentParser(description='fine-tune models with protected attributes')
    parser.add_argument('--save-dir', default='discriminatory_data/mnist01/', help='unfair_testseed_path')
    parser.add_argument('--weight-path', default='models/original/mnist01_model_onlyweight_988cf647eee71fb30938c3c2cb0df718.h5', help='vulnerable_model')
    parser.add_argument('--adv-path', default='discriminatory_data/mnist01/mnist01_model_onlyweight_988cf647eee71fb30938c3c2cb0df718_adv.npz', help='vulnerable_model')
    parser.add_argument("-n","--net_layers",default="64,32,32,16,10",type=str,help="arch by comma")


    args= parser.parse_args()

    args_arch= args.net_layers
    args_arch= [int(x) for x in args_arch.split(",")] 
    setattr(args,"net_layers",args_arch)


    main(args)
