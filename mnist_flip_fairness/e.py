import numpy as np 

labels = np.array([1,0,1,1,1,0,])

clean_predict = np.array([1,1,1,1,1,1])
success = np.array([
    [1,1,1,1,1,1],
    [0,1,1,1,1,0],
    [0,1,1,1,1,1],
    [1,1,1,1,1,1],
    ])    
expect= np.array(
     [[False,True ,False ,False, False , True],
 [ True,  True, False ,False, False, False],
 [ True , True, False, False, False , True],
 [False , True ,False, False, False , True]]
    )
if 1==1 :
    success= success.astype(np.bool)
    
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
    clean_predict= clean_predict.astype(np.bool)
    clean_predict= np.repeat( np.expand_dims(clean_predict,axis=0),len(success), axis=0)


    ret = np.logical_xor(clean_predict,pred_labels_eps_like)
    ret2= np.sum(ret,axis=0)
    unfair_count =np.count_nonzero(ret2>0) #ret2>0 #np.logical_and(ret2!=0 ,ret2!=1).mean()
    
