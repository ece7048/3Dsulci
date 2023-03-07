#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import preprocessing , regularization, create_3Dnet
from s3D.pca_gcam import *

import numpy as np
import glob
import tensorflow as tf
import skimage.transform as skTrans
#create_3D_net, regularization
from s3D.regularization import *
 
# name of weights store of main segmentation


def build_data(path1,path2,h1,w1,d1,f):
    # Each scan is resized across height, width, and depth and rescaled.
    pr=preprocessing.preprocessing(h1,w1,d1,f)
    f1 = np.array([pr.process_scan(path) for path in path1])
    f2 = np.array([pr.process_scan(path) for path in path2])

	# For the CT scans having presence of viral pneumonia
	# assign 1, for the normal ones assign 0.
    f1s = np.array([1 for _ in range(len(f1))])
    f2s = np.array([0 for _ in range(len(f2))])
    #print(path1,path2)
    # Split data in the ratio 70-30 for training and validation.
    ln1=int(0.7*len(f1))
    ln2=int(0.7*len(f2))
    x_train = np.concatenate((f1[:ln1], f2[:ln2]), axis=0)
    y_train = np.concatenate((f1s[:ln1], f2s[:ln2]), axis=0)
    x_val = np.concatenate((f1[ln1:], f2[ln2:]), axis=0)
    y_val = np.concatenate((f1s[ln1:], f2s[ln2:]), axis=0)
    print("Number of samples in train and validation are %d and %d."% (x_train.shape[0], x_val.shape[0]))

    return x_train, y_train, x_val, y_val


def load_data(x_train, y_train, x_val, y_val, bz=16):

	# Define data loaders.
	b2=x_val.shape[0]
	b,xo,yo,zo=x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]
	x_train_s=skTrans.resize(x_train, (b,xo,yo,zo,1), order=1, preserve_range=True)
	x_val_s=skTrans.resize(x_val, (b2,xo,yo,zo,1), order=1, preserve_range=True)
	x_train_t=tf.convert_to_tensor(x_train_s)
	x_val_t=tf.convert_to_tensor(x_val_s)
	y_train_t=tf.convert_to_tensor(y_train)
	y_val_t=tf.convert_to_tensor(y_val)

	train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
	print(x_train.shape)
	batch_size = bz
	# Augment the on the fly during training.
	train_dataset = (
	    train_loader.shuffle(len(x_train))
	    .map(train_preprocessing)
	    .batch(batch_size)
	    .prefetch(2)
	)
	# Only rescale.
	validation_dataset = (
	    validation_loader.shuffle(len(x_val))
	    .map(validation_preprocessing)
	    .batch(batch_size)
	    .prefetch(2)
	)

	return train_dataset, validation_dataset 

def load_images(path):
    path_s=path+'*'
    return sorted((glob.glob(path_s)))
    
def train(model="simple_3d",backbone='none',pr='off',ep=5,lr=0.0001,ls="binary_crossentropy",m1="acc",height=256,width=256,depth=100,channels=1,classes=2,name="",do=0.3,bz=16,path1="",path2="",format_file='nii',back_w='simple_3d_gpu_L_skeleton_3d_image_classification.h5'):
    weight_file=(model+name+"_3d_image_classification.h5")
    path11=[]
    path22=[]
    path11=load_images(path1)
    path22=load_images(path2)
    x_train, y_train, x_val, y_val=build_data(path1=path11,path2=path22,h1=height,w1=width,d1=depth,f=format_file)
    train_dataset, validation_dataset= load_data(x_train, y_train, x_val, y_val, bz)
    # Compile model.
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1, decay_rate=0.95, staircase=True)
    cn=create_3Dnet.create_3Dnet(model,height,width,depth,channels,classes,name="",do=0.3,backbone=backbone,paral=pr,b_w=back_w)
    model_3=cn.model_builder()
    if lr<=0.00005:
       model_3.compile(loss=ls,optimizer=tf.keras.optimizers.SGD(learning_rate=lr),metrics=[m1])
       print('fix learning rate!! ')
    else:
       model_3.compile(loss=ls,optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),metrics=[m1])
       print('exponential decrease learning rate!! ')
    if os.path.exists(weight_file):
       model_3.load_weights(weight_file)#, by_name=True, skip_mismatch=True)
       print('load file',weight_file)
    # Define callbacks.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(weight_file, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)
    # Train the model, doing validation at the end of each epoch
    epochs = ep
    model_3.fit(train_dataset,validation_data=validation_dataset,epochs=epochs,shuffle=True,verbose=2,callbacks=[checkpoint_cb, early_stopping_cb])


def shap_xai(X,model,store,model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='none',class_names=['one','two']):
    shap_preprocessing(X,model,store,model_name,height,width,depth,channels,classes,case,batch,label,rot,backbone,class_names)

def xai(image,model,store, point=[7,9],model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='none',input_n=1,casemean='mean',format_file='nii',start=0,end=0):
    
    #path11=load_images(image)
    #x_train, y_train, x_val, y_val=build_data(path1=path11,path2=path11,h1=height,w1=width,d1=depth,f=format_file)
    #train_dataset, validation_dataset= load_data(x_train, y_train, x_val, y_val, batch)
    viewer(image,model,store,point,model_name,height,width,depth,channels,classes,case,batch,label,rot,backbone,input_n,casemean,start,end)
