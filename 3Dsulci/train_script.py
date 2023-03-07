#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import train
from s3D.train import *


epoch=10
lr=0.005
loss="sparse_categorical_crossentropy"
metric="acc"
height=184
width=184
depth=64
channels=1
classes=2
name="_simple3d_L_surface"
dropout=0.3
batch=2
path_fold1='/home/mm2703/rds/hpc-work/data/sucli/surface/L/PCS/'
path_fold2='/home/mm2703/rds/hpc-work/data/sucli/surface/L/nPCS/'
format_file='nii'
model='simple_MHL'
bb='simple_3d'

train(model=model,backbone=bb,ep=epoch,lr=lr,ls=loss,m1=metric,height=height,width=width,depth=depth,channels=channels,classes=classes,name=name,do=dropout,bz=batch,path1=path_fold1,path2=path_fold2,format_file=format_file)



