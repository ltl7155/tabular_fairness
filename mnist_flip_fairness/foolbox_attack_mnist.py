#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np 
import json 


if str(tf.__version__).startswith("2."):
    tf.random.set_seed(42)
else:
    tf.random.set_random_seed(42)

class sigmoid_layer_2_softmax_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(sigmoid_layer_2_softmax_layer, self).__init__(**kwargs)

        pass 
    _name ="sigmoid2softmax"
        
    def call(self,inputs):
        # max_v=  tf.reduce_max(inputs).numpy()
        # if max_v>1 :
        #     raise Exception("only support the sogmoid's output, which should less than 1, yous input=={}".format( max_v ))
        inputs_neg = 1-inputs
        output = tf.concat([inputs_neg,inputs], axis=1)
        return output
    
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
        sigmoid_layer_2_softmax_layer(),
    ])
    model.load_weights(weight_path,by_name=True)
    #"./models/original/mnist01_model_onlyweight_988cf647eee71fb30938c3c2cb0df718.h5",by_name=True)
    return model

def get_acc(model,images,labels):
    if type(labels)!=np.ndarray :
        labels = labels.numpy() 
        
    y_logist = model.predict(images)
    if y_logist.shape[-1]==1 :
        y_logist = y_logist.numpy() if type(y_logist)!=np.ndarray else y_logist
        y_hat=( y_logist>0.5).astype(np.int32).flatten()
    else :
        y_logist =  y_logist.numpy() if type(y_logist)!=np.ndarray else y_logist
        y_hat= np.argmax(y_logist,axis=1)
        
    assert y_hat.shape==labels.shape, (y_hat.shape,labels.shape)
    acc = np.sum(labels==y_hat)/len(y_hat)
    print ("acc",acc)
    return {"pred":y_hat,"acc":acc}
def get_inconsistent_rate(model,images,labels):
    assert np.max(images)<=1 ,("expect the images's max [0,1]")

    if type(labels)!=np.ndarray :
        labels = labels.numpy() 

    y_logist = model.predict(images)
    if y_logist.shape[-1]==1 :
        y_logist = y_logist.numpy() if type(y_logist)!=np.ndarray else y_logist
        y_hat=( y_logist>0.5).astype(np.int32).flatten()
    else :
        y_logist =  y_logist.numpy() if type(y_logist)!=np.ndarray else y_logist
        y_hat= np.argmax(y_logist,axis=1)
    
    images_n = 1- images
    
    y_logist_n = model.predict(images_n)
    if y_logist_n.shape[-1]==1 :
        y_logist_n = y_logist_n.numpy() if type(y_logist_n)!=np.ndarray else y_logist_n
        y_hat_n=( y_logist_n>0.5).astype(np.int32).flatten()
    else :
        y_logist_n =  y_logist_n.numpy() if type(y_logist_n)!=np.ndarray else y_logist_n
        y_hat_n= np.argmax(y_logist_n,axis=1)
    
    assert y_hat_n.shape==y_hat.shape, (y_hat.shape,y_hat_n.shape)
    inconsistent_c = np.sum(y_hat_n!=y_hat)
    total_c = len(y_hat)
    rate = inconsistent_c/total_c
    
    
    acc = np.sum(labels==y_hat)/len(y_hat)
    acc_fairness = np.sum(labels==y_hat_n)/len(y_hat_n)
    # print (labels[:10],y_hat[:10],y_hat_n[:10])
    
    return {"acc":acc,"acc_fairness":acc_fairness,"inconsis_c":inconsistent_c,"total_c":total_c,"inconsistent_rate":rate,"y_hat":y_hat,"y_hat_n":y_hat_n} 

def get_predict_naive(model,images):
    y_logist_n = model.predict(images)
    if y_logist_n.shape[-1]==1 :
        y_logist_n = y_logist_n.numpy() if type(y_logist_n)!=np.ndarray else y_logist_n
        y_hat_n=( y_logist_n>0.5).astype(np.int32).flatten()
    else :
        y_logist_n =  y_logist_n.numpy() if type(y_logist_n)!=np.ndarray else y_logist_n
        y_hat_n= np.argmax(y_logist_n,axis=1)
    
    return y_hat_n
    

def foolbox_attack_func(model,images,labels,eps_list= (np.arange(1,11)*0.1).tolist(),
                  # is_return_advdata=False,
                  foolbox_kwargs={"bounds":(0,1),"preprocessing":{}},
                  ):
    assert 0 not in eps_list, "in order to return the unfair's count, eps_list should not include 0, your input is {}".format(eps_list)
    
    
        
    
    ret_list= {}
    fmodel: Model = TensorFlowModel(model, **foolbox_kwargs)
    attack = LinfPGD()
    if type(images)==np.ndarray :
        images = tf.convert_to_tensor(images, dtype=tf.float32)
    if type(labels)==np.ndarray :
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)

    
    raw_advs, clipped_advs, success_attack = attack(
        fmodel, images, labels, epsilons=eps_list)

    return clipped_advs,success_attack
def reverse_predict_label_from_foolboxsuccess(success_attack,labels):
    '''
    foolbox's success result always reflect the attack's sucess, but it cannot show the predict result 
    this function will use the logic to extract the predicted labels
    eg: original_label [0,1] attack_sucess [1,1]
        so the predict_label would be [1,0]
    '''
    ##############
    ''' try revert the predict label from sucess and  label
    eg : label [1,0] sucess [0,1] --> predict_label [1,1] 
    ''' 
    ##############
    ### label_eps_like=
    success= success_attack.numpy().astype(np.bool)
    
    label_eps_like = labels.numpy() if type(labels)!=np.ndarray else labels 
    if label_eps_like.ndim==1:
        label_eps_like= np.expand_dims(label_eps_like,axis=0)
        label_eps_like= np.repeat(label_eps_like,len(success), axis=0)
    
    label_eps_like= label_eps_like.astype(np.bool)
    assert label_eps_like.ndim ==2 ,label_eps_like.shape
    assert  set(np.unique(label_eps_like).tolist() ).issubset(set([0,1]) ) , "only label with 0,1 be supported"
    
    success_not = np.logical_not(success)
    pred_labels_eps_like = np.logical_xor(label_eps_like,success_not)
    pred_labels_eps_like = np.logical_not(pred_labels_eps_like)
    # print (label_eps_like,"label_eps_like\n",success,"success\n",pred_labels_eps_like,"pred_labels_eps_like\n")
    
    # print ("\n",expect,"expect")
    ### diff in clean_predict and eps_predict
    return pred_labels_eps_like


    #return unfair_metric(clean_predict=clean_predict, pred_labels_eps_like= pred_labels_eps_like,eps_list=eps_list)


def unfair_metric(clean_predict,pred_labels_eps_like,eps_list=None):
    '''
    clean_predict : 
      .shape= [1,x]
    pred_labels_eps_like:
      .shape= [len(eps),x]
    if eps_list is not None :
      assert len(eps_list) == len(pred_labels_eps_like)

    eg : if  eps_list==[0....13] assert len(eps_list)==13
          clean_predict.shape == [200,] pred_labels_eps_like.shape==[13,200]
    '''
    clean_predict_ori_len = len(clean_predict) 
    if eps_list is not None :
        assert len(pred_labels_eps_like)==len(eps_list), ("eps_list:",eps_list,"pred_labels_eps_like",pred_labels_eps_like.shape, "they should have a same length in first dim")

    assert clean_predict.ndim ==1 ,("expect clean_predict.shape==[int,]", "your are", clean_predict.shape)

    clean_predict= clean_predict.astype(np.bool)
    clean_predict= np.repeat( np.expand_dims(clean_predict,axis=0),len(pred_labels_eps_like), axis=0)


    ret = np.logical_xor(clean_predict,pred_labels_eps_like)
    ret2= np.sum(ret,axis=0)
    unfair_count =np.count_nonzero(ret2>0) #ret2>0 #np.logical_and(ret2!=0 ,ret2!=1).mean()
    
    return {"unfair_c":unfair_count,"total_c":clean_predict_ori_len}

def main(args) -> None:



    save_dir =args. save_dir
    name_prefix=os.path.splitext( os.path.basename(args.weight_path))[0]

    print ("will save as",name_prefix)



    # instantiate a model (could also be a TensorFlow or JAX model)
    model =create_model(weight_path=args.weight_path,net_layers  =args.net_layers )
    # #tf.keras.applications.ResNet50(weights="imagenet")
    pre = {}#dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(0, 1), preprocessing=pre)
    # fmodel = fmodel.transform_bounds((0, 1))
    
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
    
    gt_labels = labels


    ####
    ##check load weight
    ####
    #check the model's loading  is correct or not ?
    clean_acc = accuracy(fmodel, images, labels)
    naive_acc_info = get_acc(model=model, labels= labels ,images=images) 
    print ("test before being attacked ")
    print(f"\t clean accuracy:  {clean_acc * 100:.1f} %")
    print("\t clean(naive) accuracy:  {naive_acc:.4f} %".format( naive_acc=naive_acc_info["acc"]*100 ))


    ####
    ##get clean predict
    ####
    #clean_predict = get_predict_naive(model=model, images=images)



    # apply the attack
    attack = LinfPGD()
    # epsilons = [
    #     10,
    #     100,
    # ]
    eps_list= (np.arange(1,11)*0.1).tolist()
    print ("epsilons",)
    print ("\t", eps_list)
    
    ####
    ##attack with eps
    ####
    cliped_adv_images,_ = foolbox_attack_func(model=model,images=images,labels=labels, eps_list=eps_list)
    assert type(cliped_adv_images)==list 

    #assert_func  =lambda x: assert(type(x)==np.ndarray) 

    np_to_list = lambda x:x.tolist() if "numpy" in str(type(x)) else x
    tf_to_np = lambda x: x.numpy() if tf.is_tensor(x) else x

    save_data_list=[]
    save_info_list=[]
    img_list= []
    for eps,adv_images  in zip(eps_list, cliped_adv_images):
        inconsis_info = get_inconsistent_rate(model=model,images=adv_images,labels=gt_labels)

        pred_y_hat =tf_to_np( inconsis_info["y_hat"] )
        pred_y_hat_n =tf_to_np( inconsis_info["y_hat_n"] )
        select_idx = pred_y_hat!=pred_y_hat_n
        adv_images = tf_to_np( adv_images )
        img_list.append( adv_images [select_idx])
        assert type(adv_images) ==np.ndarray , type(adv_images)
        assert type(pred_y_hat) ==np.ndarray , type(pred_y_hat)
        assert type(pred_y_hat_n) ==np.ndarray , type(pred_y_hat_n)

        data_info ={"adv_images":adv_images,"pred_y_hat":pred_y_hat,"pred_y_hat_n":pred_y_hat_n,"eps":eps}
        save_data_list.append(data_info)


        inconsis_info_debug = {k:np_to_list(v) for k,v in inconsis_info.items() if k in ["acc","acc_fairness","inconsis_c","total_c","inconsistent_rate"]}
        save_info_list.append(inconsis_info_debug)

    adv_inconsistent= np.concatenate(img_list,axis=0)
    save_data_path = os.path.join(save_dir,"{name_prefix}_adv_inconsistent.npz".format(name_prefix=name_prefix))
    print ("data save in ",save_data_path,adv_inconsistent.shape)
    np.save(save_data_path ,adv_inconsistent)


    save_data_path = os.path.join(save_dir,"{name_prefix}_adv.npz".format(name_prefix=name_prefix))
    print ("data save in ",save_data_path)
    np.savez(save_data_path ,data=save_data_list)

    save_meta_path  = os.path.join(save_dir,"{name_prefix}_meta.json".format(name_prefix=name_prefix))
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
    parser.add_argument("-n","--net_layers",default="64,32,32,16,10",type=str,help="arch by comma")


    args= parser.parse_args()

    args_arch= args.net_layers
    args_arch= [int(x) for x in args_arch.split(",")] 
    setattr(args,"net_layers",args_arch)


    main(args)
