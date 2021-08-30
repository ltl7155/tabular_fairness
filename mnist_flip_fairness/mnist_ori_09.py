import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np 

from tensorflow import keras 

import hashlib
md5name= lambda x:hashlib.md5(x if type(x)==bytes else str(x).encode() ).hexdigest()


tf.random.set_seed(42)
np.random.seed(42)

import pre_mnist_01





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
    if images.ndim ==2 :
        assert images.shape[-1]==28*28 ,images.shape
        images=images.reshape(len(images),28,28)
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

def predict_metric(model,x_data,y_data,show_text="pred",label_choices=[0,9],save_dir="./images/"):
    from sklearn.metrics import classification_report
    import pandas as pd 
    import io 
    
    os.makedirs(save_dir,exist_ok=True)
    
    y_hat_logist = model.predict(x_data)

    if y_hat_logist.ndim==2 and y_hat_logist.shape[-1]==1:# sigmoid
        y_hat_pred =(y_hat_logist > 0.5).astype(int).flatten()
        
    else :
        y_hat_pred = np.argmax(y_hat_logist,axis=1)
    
    ##save debug 
    display_images(
        images=x_data[:32],
        titles=["label_{}".format(x) for x in y_data[:32]],
        save_filename=os.path.join(save_dir,"{}.jpg".format(show_text)),
        )
    ## report
    info_dict= classification_report(
        y_true=y_data, y_pred=y_hat_pred,
        output_dict=True)
    del info_dict["macro avg"]
    del info_dict["weighted avg"]
    
    df = pd.DataFrame(info_dict).T
    df.rename(index={'0':str(label_choices[0]),'1':str(label_choices[-1])},inplace=True)
    # df .columns = ["precision","recall",  "f1-score"   "support"] 
    ret_info = df[["precision"]]
    print ("====="*8,show_text)
    print (ret_info)
    print ("\n\n")
    return {"y_pred":y_hat_pred,"reporter": ret_info.to_dict()}


if __name__=="__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("-r","--rate",type=float,default=0.2)
    parser.add_argument("-l","--labels",choices=list(map(str, range(0,10))),default=[0,9],nargs="+")
    parser.add_argument("-e","--epochs",default=5,type=int)
    parser.add_argument("-n","--net_layers",default="64,32,32,16,10",type=str,help="arch by comma")
    
    args = parser.parse_args()
    print (args)
    args_info_dict = vars(args)
    assert type(args_info_dict)==dict 

    ##########
    ##### config 
    ##########
    flip_rate=args.rate
    # label_choices=[0,9]
    label_choices=args.labels
    label_choices= [int(x) for x in label_choices]
    
    ##load 
    dataset_dict = pre_mnist_01.func_call(flip_rate=flip_rate,label_choices= label_choices)
    x_train = dataset_dict["X_train"]
    y_train = dataset_dict["y_train"]
    x_test = dataset_dict["x_val"]
    y_test = dataset_dict["y_val"]
    
    # display_images(
    #     images=x_train[:32],
    #     titles=["label_{}".format(x) for x in y_train[:32]],
    #     save_filename="origin_train.jpg",
    #     )
    #
    # ## print ...
    #
    #
    # ##debug 
    # display_images(
    #     images=x_train[:32],
    #     titles=["label_{}".format(x) for x in y_train[:32]],
    #     save_filename="flip_and_noflip_train.jpg",
    #     )
    #

    
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(64, activation="relu"),
    #     tf.keras.layers.Dense(32, activation="relu"),
    #     tf.keras.layers.Dense(16, activation="relu"),
    #     tf.keras.layers.Dense(1, activation="sigmoid")
    # ])
    args_arch= args.net_layers
    args_arch= [int(x) for x in args_arch.split(",")] 
    archs = [tf.keras.layers.Dense(x, activation="relu",name=f"layer_{ii}")
             for ii,x in enumerate(args_arch) 
             ]
    
    model = tf.keras.models.Sequential([
        keras.Input(shape=(28*28),name="input"),
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        *archs,
        tf.keras.layers.Dense(1, activation="sigmoid",name="output")
        ])
    
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )
    model.compile(loss="binary_crossentropy", 
                  optimizer=tf.keras.optimizers.Adam(0.001), 
                  metrics=["accuracy"]
                  )
    model.summary()
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test,y_test),
        epochs=int(args.epochs),
        verbose=0,
    )
    
    
    size_t_total = len(x_test)
    
    
    ret_info_ori = predict_metric(model=model,x_data=x_test,y_data=y_test,show_text="test_all(size={})_(rate={})".format(size_t_total,flip_rate),label_choices=label_choices)
    y_pred_ori= ret_info_ori["y_pred"]
    
    ret_info_flip = predict_metric(model=model,x_data=1-x_test,y_data=y_test,show_text="fairness_test_all(size={})_(rate={})".format(size_t_total,flip_rate),label_choices=label_choices)
    y_pred_flp= ret_info_flip["y_pred"]

    print ("total inconsistent rate {}/{}".format( np.sum(y_pred_ori!=y_pred_flp), len(y_pred_flp)  ) )
    
    serise_no = md5name(str(args_info_dict))
    print (str(args_info_dict),"str(args_info_dict)",serise_no)
    
    args_info_dict.update({"ori_reporter":ret_info_ori["reporter"], "flip_reporter":ret_info_flip["reporter"], })
    
    model_name = 'models/original/mnist01_model_{}.h5'.format(serise_no)
    keras.models.save_model(model, model_name)
    # model.save_weights( model_name)
    w_name = 'models/original/mnist01_model_onlyweight_{}.h5'.format(serise_no)
    model.save_weights( w_name)

    modelinfo_name = 'models/original/mnist01_model_{}_meta.json'.format(serise_no)
    import json
    with open(modelinfo_name,"w") as f :
        json.dump(fp=f,obj=args_info_dict,indent=2) 
