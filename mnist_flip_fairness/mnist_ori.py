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
    '''
    total 6w -> 
    in order to causly flip the some image, donot load keras.data.load_mnist()
    '''
    flip_size= 0 if flip_rate is None else  round(float(flip_rate)*len(x_train))
    flip_size_test= 0 if flip_rate is None else  round(float(flip_rate)*len(x_test))
    assert len(x_train)>= flip_size
    assert len(x_test)>= flip_size_test
    
    np.random.seed(random_state)
    index_train=  np.random.permutation(np.arange(len(x_train) ) )
    
    ## flip a white bg into black bg
    index_train_flip = index_train[:flip_size]
    index_train_noflip = index_train[flip_size:]
    
    x_train_fliped = x_train[index_train_flip] 
    x_train_fliped = func_flip_background(x_train_fliped)
    
    x_train[index_train_flip]=x_train_fliped
    
    ##
    ####test 
    ##
    index_test=  np.random.permutation(np.arange(len(x_test) ) )
    index_test_flip = index_test[:flip_size_test]
    index_test_noflip = index_test[flip_size_test:]

    x_test_fliped = x_test[index_test_flip] 
    x_test_fliped = func_flip_background(x_test_fliped)

    x_test[index_test_flip]=x_test_fliped

    assert len(x_train)== len(y_train) and  len(x_test) == len(y_test),\
        ("train.len",len(x_train),"train.y.len", len(y_train) ,"test.x.len" , len(x_test) ,"test.y.len", len(y_test))
    
    return (x_train,y_train,index_train_flip,index_train_noflip ), (x_test,y_test,index_test_flip,index_test_noflip) 

def display_images(images, titles=None, cols=8, \
                cmap="gray", \
                norm=None,\
                   interpolation=None,save_filename="test.jpg"):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    #cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    import matplotlib.pyplot as plt
    if "float" in str(images .dtype) :
        images = images.copy()
        images = (images*255).astype(np.uint8)
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(13, 13 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), 
                    cmap=cmap,
                    norm=norm, 
                    interpolation=interpolation
                   )
        i += 1
    plt.savefig(save_filename) 

def predict_metric(model,x_data,y_data,show_text="pred"):
    from sklearn.metrics import classification_report
    import pandas as pd 
    import io 
 
    y_hat_logist = model.predict(x_data)
    y_hat_pred = np.argmax(y_hat_logist,axis=1)
    
    ##save debug 
    display_images(
        images=x_data[:32],
        titles=["label_{}".format(x) for x in y_data[:32]],
        save_filename="{}.jpg".format(show_text),
        )
    ## report
    info_dict= classification_report(
        y_true=y_data, y_pred=y_hat_pred,
        output_dict=True)
    del info_dict["macro avg"]
    del info_dict["weighted avg"]
    
    df = pd.DataFrame(info_dict).T
    # df .columns = ["precision","recall",  "f1-score"   "support"] 
    ret_info = df[["precision"]]
    print ("====="*8,show_text)
    print (ret_info)
    print ("\n\n")
    return ret_info 
##load 
(x_train,y_train),(x_test,y_test) = \
    keras.datasets.mnist.load_data()
x_train  = normlize(x_train)
x_test  = normlize(x_test)

display_images(
    images=x_train[:32],
    titles=["label_{}".format(x) for x in y_train[:32]],
    save_filename="origin_train.jpg",
    )

## flip with a split rate
mnist_keras_dataset= load_mnist_from_numpy(x_train=x_train,
           y_train=y_train,   x_test=x_test,    y_test=y_test)

(x_train,y_train,index_train_flip,index_train_noflip ), \
    (x_test,y_test,index_test_flip,index_test_noflip)  = mnist_keras_dataset

##debug 
display_images(
    images=x_train[:32],
    titles=["label_{}".format(x) for x in y_train[:32]],
    save_filename="flip_and_noflip_train.jpg",
    )


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(32,activation='relu'),
  tf.keras.layers.Dense(16,activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test,y_test),
    epochs=10,
)


size_t_f= len(index_test_flip)
size_t_nf= len(index_test_noflip)
size_t_total = len(x_test)

predict_metric(model=model,x_data=x_test[index_test_flip],y_data=y_test[index_test_flip],show_text="test_flip(size={})".format(size_t_f))
predict_metric(model=model,x_data=x_test[index_test_noflip],y_data=y_test[index_test_noflip],show_text="test_noflip(size={})".format(size_t_nf))
predict_metric(model=model,x_data=1-x_test[index_test_flip],y_data=y_test[index_test_flip],show_text="fairness_test_flip(size={})".format(size_t_f))
predict_metric(model=model,x_data=1-x_test[index_test_noflip],y_data=y_test[index_test_noflip],show_text="fairness_test_noflip(size={})".format(size_t_nf))

predict_metric(model=model,x_data=x_test,y_data=y_test,show_text="test_all(size={})".format(size_t_total))
predict_metric(model=model,x_data=1-x_test,y_data=y_test,show_text="fairness_test_all(size={})".format(size_t_total))



