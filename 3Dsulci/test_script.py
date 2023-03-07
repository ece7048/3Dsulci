#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import test
from s3D.test import *


lr=0.001
loss="sparse_categorical_crossentropy"
metric="acc"
height=180
width=180
depth=60
channels=1
classes=2
name="_cpu_L_skeleton"
dropout=0.3
batch=2
path_fold1='/home/mm2703/rds/hpc-work/data/sucli/skeleton/L/PCS/'
path_fold2='/home/mm2703/rds/hpc-work/data/sucli/skeleton/L/nPCS/'
format_file='nii'
model='simple_MHL'
store_path='/home/mm2703/rds/hpc-work/data/sucli/skeleton/L/'
label=["PCS","nPCS"]
bb='none'

test(model=model,lr=lr,ls=loss,m1=metric,height=height,width=width,depth=depth,channels=channels,classes=classes,name=name,do=dropout,bz=batch,path1=path_fold1,path2=path_fold2,format_file=format_file,store_path=store_path,labels1=label)
