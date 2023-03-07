#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from s3D import train
from s3D.train import *

# name of weights store of main segmentation

image= '/home/mm2703/rds/hpc-work/data/sucli/surface/R/PCS/'      
model="/home/mm2703/rds/hpc-work/code/s3D/test/simple_3d_cpu_surface_3d_image_classification.h5"
store= "/home/mm2703/rds/hpc-work/data/sucli/3DXAI/R/PCS/"
point=[1,14]

xai(image,model,store,point,model_name='simple_3d',height=200,width=200,depth=155,channels=1,classes=2,case='3d_image_classification',batch=1,label=[1,0],rot=180,backbone="none",casemean='pca')

image1= '/home/mm2703/rds/hpc-work/data/sucli/surface/R/nPCS/'
model1="/home/mm2703/rds/hpe-work/code/s3D/test/simple_3d_cpu_surface_3d_image_classification.h5"
store1= "/home/mm2703/rds/hpc-work/data/sucli/3DXAI/R/nPCS/"
point1=[1,14]

xai(image1,model1,store1,point1,model_name='simple_3d',height=200,width=200,depth=155,channels=1,classes=2,case='3d_image_classification',batch=1,label=[0,1],rot=180,backbone="none",casemean='pca')


