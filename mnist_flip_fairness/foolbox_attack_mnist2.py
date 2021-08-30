#!/usr/bin/env python3
import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import LinfPGD
import numpy as np 

gpus = tf.config.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
            ]
        )
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



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
    
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu",name="layer_0" ),
        tf.keras.layers.Dense(32, activation="relu",name="layer_1"),
        tf.keras.layers.Dense(32, activation="relu",name="layer_2"),
        tf.keras.layers.Dense(16, activation="relu",name="layer_3"),
        tf.keras.layers.Dense(1, activation="sigmoid",name="output"),
        sigmoid_layer_2_softmax_layer(),
    ])
    model.load_weights("./models/original/mnist01_model_w_c14fbf8c633c593f029630ad41bd2c7b.h5",by_name=True)
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
    
    return {"acc":acc,"acc_fairness":acc_fairness,"inconsis_c":inconsistent_c,"total_c":total_c,"inconsistent_rate":rate} 


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    model =create_model() #tf.keras.applications.ResNet50(weights="imagenet")
    pre = {}#dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(0, 1), preprocessing=pre)
    # fmodel = fmodel.transform_bounds((0, 1))


    ##test acc 
    (_,_),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    y_test_index09 =np.logical_or( y_test==0,y_test==9) 
    x_test = (x_test[y_test_index09].astype(np.float))/255.
    y_test = y_test[y_test_index09]
    y_test[y_test==9]=1
    
    # #print ("x_test",np.min(x_test),np.max(x_test))
    # y_test_hat = model(x_test)
    # print ("y_test_hat",y_test_hat.shape,"--->")
    # y_test_hat = y_test_hat.numpy()
    # # y_test_hat = (y_test_hat>0.5).astype(np.int32).flatten()
    # y_test_hat = np.argmax(y_test_hat,axis=-1)# (y_test_hat>0.5).astype(np.int32).flatten()
    #
    # acc=  np.sum(y_test_hat==y_test) /  len(y_test)
    # print ("acc",acc)
    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels =tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.int64)
    # ep.astensors(x_test,y_test )  
    # ddd = samples(fmodel, dataset="mnist", batchsize=16)
    # d1,d2 = ddd [:2]
    # print (type(d1),type(d2),"d1,d2")
    # print (d1.dtype,d2.dtype,"d1,d2")
    # print (type(images),type(labels),"images,labels")
    # exit()
    clean_acc = accuracy(fmodel, images, labels)
    naive_acc_info = get_acc(model=model, labels= labels ,images=images) 
    print ("test before being attacked ")
    print(f"\t clean accuracy:  {clean_acc * 100:.1f} %")
    print("\t clean(naive) accuracy:  {naive_acc:.4f} %".format( naive_acc=naive_acc_info["acc"]*100 ))

    # apply the attack
    attack = LinfPGD()
    # epsilons = [
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     10,
    #     30,
    #     100,
    # ]
    epsilons = np.array  (range(0,11,1)) * 0.1
    epsilons = epsilons.tolist()
    print ("epsilons",)
    print ("\t", epsilons)
    
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     10,
    #     30,
    #     100,
    # ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    
    print ("test after being attacked ")
    for eps,one_adv in zip(epsilons,clipped_advs):
        print ("===="*8)
        naive_info = get_acc(model=model,images=one_adv,labels=y_test)
        clean_acc = accuracy(fmodel, one_adv, labels)

        incon_info = get_inconsistent_rate(model=model,images=one_adv,labels=y_test)
        print ("consistent info ", incon_info)
        # print(f"eps: {eps} accuracy:  {clean_acc * 100:.1f} %")
        print("eps: {eps} accuracy:  {naive_acc:.4f} %".format(eps=eps,naive_acc=naive_info["acc"]*100))
        np.savez("./mnist_pgd_(eps={})_(acc={}).npz".format(eps,naive_info["acc"] ),
                 adv_data= one_adv.numpy(),
                 pred_labels= naive_info["pred"] if type(naive_info["pred"])==np.ndarray else naive_info["pred"].numpy(),
                 ground_truth = labels if type(labels) ==np.ndarray else labels.numpy(),
                 )
        print ("\n")
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    # robust_accuracy = 1 - success.numpy().mean(axis=-1)
    # print("robust accuracy for perturbations with")
    # for eps, acc in zip(epsilons, robust_accuracy):
    #     print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    # print()
    # print("we can also manually check this:")
    # print()
    # print("robust accuracy for perturbations with")
    # for eps, advs_ in zip(epsilons, clipped_advs):
    #     acc2 = accuracy(fmodel, advs_, labels)
    #     print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
    #     print("    perturbation sizes:")
    #     # print ("advs_","images",type(advs_),type(images))
    #
    #     advs_,images = ep.astensors(advs_.numpy(),images.numpy() )
    #
    #     # print ("advs_","images",advs_.shape,images.shape)
    #
    #     perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2)).numpy()
    #     print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
    #     if acc2 == 0:
    #         break


if __name__ == "__main__":
    main()