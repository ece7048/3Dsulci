#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import preprocessing , regularization, create_3Dnet
import numpy as np
import glob
import tensorflow as tf
from s3D import pca_gcam
from s3D.pca_gcam import *
from tensorflow.keras.utils import to_categorical
#create_3D_net, regularization
from s3D.regularization import *

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, roc_auc_score, f1_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
#from sklearn.metrics import plot_confusion_matrix
from itertools import cycle
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
    print(path1,path2)
    # Split data in the ratio 70-30 for training and validation.
    ln1=int(0.7*len(f1))
    ln2=int(0.7*len(f2))
    x_t = np.concatenate((f1[:ln1], f2[:ln2]), axis=0)
    y_t = np.concatenate((f1s[:ln1], f2s[:ln2]), axis=0)
    x_v = np.concatenate((f1[ln1:], f2[ln2:]), axis=0)
    y_v = np.concatenate((f1s[ln1:], f2s[ln2:]), axis=0)
    print("Number of samples in train and validation are %d and %d."% (x_t.shape[0], x_v.shape[0]))

    return x_t, y_t, x_v, y_v

def load_best_callback(model_structure,X,Y):
    #model_structure.load_weights(self.store_model_path + '/weights_%s_%s.h5' %(weight_name,case))
    fpr,tpr,aucp=dict(),dict(),dict()
    thresholds=0
    y_pred_keras = model_structure.predict(X) #.ravel()
    print(np.array(y_pred_keras.shape))
    print('before the discretization')
    y_pred=np.array(y_pred_keras)
    y_pred_keras=np.array(y_pred_keras)
    #y_pred_keras=np.absolute(np.array(y_pred_keras)/np.max(y_pred_keras))
    #y_pred_keras[np.arange(len(y_pred_keras)), y_pred_keras.argmax(1)] = 1
    print((y_pred_keras[4,:],np.array(y_pred_keras).shape[0]))
    length=np.array(y_pred_keras).shape[0]
    for i in range(length):
        y_max=np.max(y_pred_keras[i,:])
        y_pred_keras[i]=np.where(y_pred_keras[i]==y_max, 1,0 ) #change to predict continious
    print('after the discretization')
    print((y_pred_keras[4,:]))
    print(y_pred_keras.shape,y_pred.shape,Y.shape)
    for i in range(np.array(y_pred_keras.shape[1])):
        y1=Y[:,i]
        y2=y_pred[:,i]
        print(y1.shape)
        print(y2)
        print(y1)        
        fpr[i], tpr[i], thresholds = roc_curve(y1,y2,pos_label=1)
        print(fpr)
        aucp[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, aucp, y_pred_keras   


def load_images(path):
    return sorted((glob.glob(path+"*")))
    
def test(model="simple_3d",backbone='none',bw='none',lr=0.0001,ls="binary_crossentropy",m1="acc",height=256,width=256,depth=100,channels=1,classes=2,name="",do=0.3,bz=16,path1="",path2="",format_file='nii',store_path='',labels1=["PCS","nPCS"]):
    
    path11=load_images(path1)
    path22=load_images(path2)
    x_train, y_train, x_val, y_val=build_data(path1=path11,path2=path22,h1=height,w1=width,d1=depth,f=format_file)
    # Compile model.
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
    cn=create_3Dnet.create_3Dnet(model,height,width,depth,channels,classes,name="",do=0.3,backbone=backbone,b_w=bw)
    model_3=cn.model_builder()
    #model_3.compile(loss=ls,optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),metrics=[m1])
    # call metrics compute
    weight_file=(model+name+"_3d_image_classification.h5")
    print(weight_file)
    if os.path.exists(weight_file):
        model_3.load_weights(weight_file) #,by_name=True, skip_mismatch=True)
    else:
        print('WARNING: no saved weights!!! run with untraining model !!!')
    store_path2=store_path+'/val/'
    if not os.path.exists(store_path2):
        os.mkdir(store_path2)
    test_result(x_train, y_train,model_3, model,store_path,labels1)
    test_result(x_val, y_val,model_3, model,store_path2,labels1)

def test_result(X,Y,model_structure, model_name,spath,labels1):

    '''
       Test model to create the main segmentation model of image
    '''
    classes_name=labels1
    classes_num=len(labels1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    print(len(classes_name))
    print(classes_name)
    print(model_name)
    Yc=to_categorical(np.asarray(Y),(classes_num))
    mainY2=Yc
    mainXtest=X
    print(Yc.shape)
    for o in range(1):
        plt.figure()
        fpr, tpr, roc_auc, y_t=load_best_callback(model_structure,mainXtest,mainY2)
        class_number=range(classes_num)
        for i, color in zip(range(classes_num), colors):
            print(roc_auc[i])
            plt.plot(fpr[i], tpr[i],label='ROC curve area = %f of  %s class '  %( roc_auc[i], classes_name[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve characteristics of %s' %model_name)
        plt.legend(loc="lower right")
        plt.savefig(spath+'ROC_curve_%s.png' % model_name)
        plt.close()
    #    fig, ax = plt.subplots()
        
     #   RocCurveDisplay.from_estimator(model_structure, mainXtest, mainY2, ax = ax)
     #   plt.savefig(spath+'ROC_curve_estimator_%s.png' % model_name[o])
     #   plt.close()

        fig2, ax2 = plt.subplots()
        metrics.RocCurveDisplay.from_predictions(mainY2[:,0],y_t[:,0],ax=ax2,name="deep learning predictions")
        plt.savefig(spath+'ROC_curve_prediction_%s.png' % model_name)
        plt.close()
#        plt.figure()
#        fig, ax = plt.subplots(16, 16)
        print(y_t.shape,mainY2.shape)
        print(y_t[1,:],mainY2[1,:])
        mainY2=np.reshape(mainY2,(mainY2.shape[0],mainY2.shape[1]))
        y_t=np.asarray(y_t)
        mainY2=np.asarray(mainY2)
        y_t=np.reshape(y_t,(mainY2.shape[0],mainY2.shape[1]))
        print(y_t.shape,mainY2.shape)
         #vis_arr=np.asarray(multilabel_confusion_matrix((y_t),mainY2))
         #print(vis_arr.shape)
         #print(vis_arr)
         #for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, classes_name):
         #    print(cfs_matrix.shape)
         #fig.tight_layout()
         #plt.savefig(spath+'Coeff_Metr_%s.png' % model_name[o])
         #print(y_t.shape,mainY2.shape)
        y_t1=y_t #.astype(int)
        mainY21=mainY2 #.astype(int)
        print(y_t1.shape,mainY21.shape)
        try:
            ras=roc_auc_score(y_t1,mainY21,average='macro', multi_class='ovr')
        except ValueError:
            ras=0
            pass

        try:
            rasq=roc_auc_score(y_t1,mainY21,average='micro')
        except ValueError:
            rasq=0
            pass
                            
        try:
            rasw=roc_auc_score(y_t1,mainY21,average='weighted')
        except ValueError:
            rasw=0
            pass
        try:
            rase=roc_auc_score(y_t1,mainY21,average='samples', multi_class='ovr')
        except ValueError:
            rase=0
            pass
        print("The results of class")
        print ("are:")

        try:
            f1s=f1_score(y_t1,mainY21,average='macro')
        except ValueError:
            f1s=0
            pass
        try:
            f1sq=f1_score(y_t1,mainY21,average='micro')
        except ValueError:
            f1sq=0
            pass
        try:
            f1sw=f1_score(y_t1,mainY21,average='weighted')
        except ValueError:
            f1sw=0
            pass
        try:
            f1se=f1_score(y_t1,mainY21,average='samples')
        except ValueError:
            f1se=0
            pass
        try:
            pr=precision_score(y_t1,mainY21, average='samples')
        except ValueError:
            pr=0
            pass
        try:
            re=recall_score(y_t1,mainY21, average='samples')
        except ValueError:
            re=0
            pass

        print('RAS macro:')
        print(ras)
        print('RAS micro:')
        print(rasq)
        print('RAS weighted:')
        print(rasw)
        print('RAS sample:')
        print(rase)
        print('F1: macro')
        print(f1s)
        print('F1 micro:')
        print(f1sq)
        print('F1 weighted:')
        print(f1sw)
        print('F1 samples:')
        print(f1se)
        print('Prec:')
        print(pr)
        print('Recall:')
        print(re)
        y_pred_str=map(str,y_t)
        ns="/y_pred_out"+"%s.csv" %(model_name[o])
        with open(spath+ns, mode='w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            y_pred_str=zip([map(str,y_t[:,u]) for u in range((np.array(y_t).shape[1]))])
            print("y_pred prediction:")
            for row in y_pred_str:
                for t in row:
                    employee_writer.writerow(t)
                    print(t)
                                                        
        y_real_str=map(str,mainY2)
        ns2="/y_real_out"+"%s.csv" %(model_name[o])
        with open(spath+ns2, mode='w') as employee_file2:
            employee_writer2 = csv.writer(employee_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            y_r_str=zip([map(str,mainY2[:,u]) for u in range((np.array(mainY2).shape[1]))])
            print("y_r_str prediction:")
            for row2 in y_r_str:
                 for s in row2:
                    employee_writer2.writerow(s)
                    print(s)

