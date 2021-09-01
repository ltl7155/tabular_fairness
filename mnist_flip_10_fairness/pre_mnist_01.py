import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


func_flip_background=  lambda x:1-x
normlize=  lambda x:(x.astype(np.float32)/255.)


def load_mnist_from_numpy(x_train,
                          y_train,
                          x_test,y_test,
                          flip_rate=0.2,\
                          random_state=32):
    #tf.random.set_seed(int(random_state))
    np.random.seed(int(random_state))

    '''
    total 6w -> 
    in order to causly flip the some image, donot load keras.data.load_mnist()
    '''
    x_train= x_train.reshape(len(x_train),-1)
    x_test= x_test.reshape(len(x_test),-1)
    
    
    flip_size= 0 if flip_rate is None else  round(float(flip_rate)*len(x_train))
    flip_size_test= 0 if flip_rate is None else  round(float(flip_rate)*len(x_test))
    assert len(x_train)>= flip_size
    assert len(x_test)>= flip_size_test
    
    index_train=  np.random.permutation(np.arange(len(x_train) ) )
    
     
    ## flip a white bg into black bg
    index_train_flip = index_train[:flip_size]
    index_train_noflip = index_train[flip_size:]
    
    x_train_fliped = x_train[index_train_flip] 
    x_train_fliped = func_flip_background(x_train_fliped)
    
    x_train[index_train_flip]=x_train_fliped
    
    y_train_protected_one_attr=np.zeros(len(index_train),dtype=np.int)
    y_train_protected_one_attr[index_train_flip]=1
    ##
    ####val 
    ##
    x_val= x_test.copy()
    y_val= y_test.copy()
    index_val=  np.random.permutation(np.arange(len(x_val) ) )
    index_val_flip = index_val[:flip_size_test]
    index_val_noflip = index_val[flip_size_test:]
    
    x_val_fliped = x_val[index_val_flip] 
    x_val_fliped = func_flip_background(x_val_fliped)
    
    x_val[index_val_flip]=x_val_fliped
    
    y_val_protected_one_attr = np.zeros(len(index_val),dtype=np.int)
    y_val_protected_one_attr[index_val_flip]=1


    assert len(x_train)== len(y_train) and  len(x_test) == len(y_test),\
        ("train.len",len(x_train),"train.y.len", len(y_train) ,"test.x.len" , len(x_test) ,"test.y.len", len(y_test))
    
    return (x_train.view(),y_train,y_train_protected_one_attr ), \
        (x_val,y_val,y_val_protected_one_attr) , (x_test,y_test,None) 




normlize=  lambda x:(x.astype(np.float32)/255.)
def func_call(flip_rate=0.2,label_choices=[0,9]):
    ##load 
    (x_train,y_train),(x_test,y_test) = \
        keras.datasets.mnist.load_data()
    x_train  = normlize(x_train)
    x_test  = normlize(x_test)
#    index_09=np.logical_or( y_train==label_choices[0] , y_train==label_choices[-1])
#    index_09_test=np.logical_or( y_test==label_choices[0] , y_test==label_choices[-1])
    
#    x_train = x_train[index_09]
#    y_train = y_train[index_09]
#    y_train [y_train==label_choices[0]]=0 
#    y_train [y_train==label_choices[-1]]=1 
#    x_test = x_test[index_09_test]
#    y_test = y_test[index_09_test]
#    y_test [y_test==label_choices[0]]=0 
#    y_test [y_test==label_choices[-1]]=1 

    ## flip with a split rate
    mnist_keras_dataset= load_mnist_from_numpy(x_train=x_train,
               y_train=y_train,   x_test=x_test,    y_test=y_test,
               flip_rate=flip_rate)
    
    (x_train,y_train,y_train_protected_attr ), \
    (x_val,y_val,y_val_protected_attr ), \
        (x_test,y_test,_)  = mnist_keras_dataset
    
    print ("xtrain",x_train.shape,y_train.shape,"flip_size",np.sum(y_train_protected_attr) )
    print ("xtest",x_test.shape,y_test.shape)

    assert len(y_train_protected_attr)==len(x_train)
    
    return {
        "X_train":x_train,
        "y_train":y_train,
        "x_val":x_val,
        "y_val":y_val,
        
        "x_test":x_test,
        "y_test":y_test,
        "constraint":np.array([[0,1]]),
        "protected_attribs":None,## because you cannot extract the value by protected_attribs from x_train 
        "y_train_protected":y_train_protected_attr,
        "y_val_protected":y_val_protected_attr,
        }

    
