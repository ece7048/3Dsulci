#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import train
from s3D.train import *

# name of weights store of main segmentation

#image= '/home/mm2703/data/sucli/skeleton/R/PCS/'      
#model="/home/mm2703/code/s3D/test/simple_3d_3d_image_classification.h5"
#store= "/home/mm2703/data/sucli/3DXAI/PCS/R/"
#point=[1,14]
#end=10
# run train of main segmentation
#xai(image,model,store,point,model_name='simple_3d',height=200,width=200,depth=150,channels=1,classes=2,case='3d_image_classification',batch=1,label=[1,0],rot=180,backbone="none",casemean='pca')

image1= '/home/mm2703/rds/hpc-work/data/sucli/surface/R/'
model1="/home/mm2703/rds/hpc-work/code/s3D/test/simple_MHL_simple3d_R_surface_3d_image_classification.h5"
store1= "/home/mm2703/rds/hpc-work/data/sucli/3DXAI/R/surface/"
# run train of main segmentation
bk='simple_3d'
shap_xai(X=image1,model=model1,store=store1,model_name='simple_MHL',height=184,width=184,depth=64,channels=1,classes=2,case='3d_image_classification',batch=1,label=[0,1],rot=180,backbone=bk,class_names=["PCS","nPCS"])
