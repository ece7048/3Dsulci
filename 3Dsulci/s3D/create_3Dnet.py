#Author: Michail Mamalakis
#Version: 0.1
#Licence:MIT
#email:mm2703@cam.ac.uk

#an extention including Resnet3DBuilder from https://github.com/JihongJu/keras-resnet3d
# pip install git+https://github.com/JihongJu/keras-resnet3d.git

from __future__ import division, print_function
import os
import zipfile
import numpy as np
import tensorflow as tf
from resnet3d import Resnet3DBuilder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, MultiHeadAttention, Input, Conv2D, Concatenate, MaxPooling2D, AveragePooling2D, AveragePooling1D, Dense, Flatten, Reshape, Activation, Dropout, Dense
from tensorflow.keras.models import Model
# name of weights store of main segmentation


class create_3Dnet:

	def __init__(self, model,height,width,depth,channels,classes,name="",do=0.3, path="/home/mm2703/code/s3D/test/",backbone="simple_3d",paral='off',b_w='simple_3d_gpu_L_skeleton_3d_image_classification.h5'):
		self.model=model
		self.height=height
		self.width=width
		self.depth=depth
		self.channels=channels
		self.classes=classes
		self.path=path
		self.name=name
		self.do=do
		self.backbone=backbone                
		self.backb_w=b_w
		self.par=paral
	def model_builder(self):
		if self.model=="simple_3d":
			init_model=self.simple_3d()
			model=self.MLP(init_model)	
		elif self.model=='simple_MHL':
			init_model=self.tune_MHL(backbone=self.backbone,name=self.name,store_model=self.path,parallel=self.par)

			model_file=str(self.path + "/"+self.backb_w)
			if os.path.exists(model_file):
				print(model_file)
				init_model.load_weights(model_file,by_name=True, skip_mismatch=True)
			model=self.MLP(init_model)
		elif self.model=='3D_resnet_50':
			model = Resnet3DBuilder.build_resnet_50((self.height, self.width, self.depth, self.channels),self.classes )
		else:
			print("no model is given")
		return model


	def simple_3d(self,backbone_use='off'):
		"""Build a 3D convolutional neural network model."""

		inputs = keras.Input((self.width, self.height, self.depth, 1))

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
		x = layers.MaxPool3D(pool_size=2)(x)
		x = layers.BatchNormalization()(x)

		x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)
		x = layers.BatchNormalization()(x)

		x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)
		x = layers.BatchNormalization()(x)

		x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
		x = layers.MaxPool3D(pool_size=2)(x)
		x = layers.BatchNormalization()(x)

		x = layers.GlobalAveragePooling3D()(x)
		x = layers.Dense(units=512, activation="relu")(x)
		x = layers.Dropout(self.do)(x)
		if backbone_use=='off':
			outputs = layers.Dense(units=1024, activation="softmax")(x)
		else:
			outputs = layers.Dense(units=(self.height*self.width), activation="softmax")(x)
		# Define the model.
		model = Model(inputs, outputs, name="3dcnn")
		return model

	def tune_MHL(self,backbone="none",name="",attention="_3d_image_classification",store_model="",parallel='off'):
		inputs=keras.Input((self.width,self.height,self.depth,1))
		if backbone=="none":
			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)
			print("case M-Head attention MHL ")
			x = layers.GlobalAveragePooling3D()(x)
			rc = layers.Dense(units=(self.height*self.width), activation="relu")(x)
			
		elif backbone=="simple_3d_tune":
			Smodel=self.simple_3d('on')
			model_file=str(store_model + "/"+self.backb_w)
			print(model_file)
			if os.path.exists(model_file):
				Smodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load denset weights')
			rc=Smodel(inputs)
			#Rc=Smodel.output 

		elif backbone=="simple_3d":
			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			x = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			rc1 = layers.BatchNormalization()(x)

			x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
			x = layers.MaxPool3D(pool_size=2)(x)
			rc2 = layers.BatchNormalization()(x)

			x = layers.GlobalAveragePooling3D()(x)
			x = layers.Dense(units=512, activation="relu")(x)
			rc = layers.Dense(units=(self.height*self.width), activation="relu")(x)
			print("case M-Head attention simple model ")
		else:
			print("No none backbone network try resnet50, densenet121, or none!")
		Rdo=layers.Flatten(name='flatten_tunedR')(rc)
		if parallel=='on':
			Rd1=layers.Flatten(name='flatten_tunedR1')(rc1)
			Rd2=layers.Flatten(name='flatten_tunedR2')(rc2)
			R=layers.MultiHeadAttention(num_heads=3,key_dim=self.height,attention_axes=(1))(Rd1,Rd2,Rdo)
			Rd=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(R,R)
		else:
			Rd=Rdo
		rgb=layers.MultiHeadAttention(num_heads=2,key_dim=self.height,attention_axes=(1))(Rd,Rd)
		rgb1=layers.Reshape([self.height,self.width,1,1])(rgb)
		#rgb2=Reshape([self.height,self.width,1,1])(Rd)
		#rgbc=Concatenate(axis=3)([rgb1,rgb2])
		#r=Reshape([self.height,self.width,2,1])(rgbc)
		RCC=layers.Conv3D(filters=self.depth, kernel_size=1, activation="relu")(rgb1)
		rgx=layers.Reshape([self.height,self.width,self.depth,1])(RCC)
		x = layers.GlobalAveragePooling3D()(rgx)
		Rdx=layers.Flatten(name='flatten_tunedRx')(x)  
		rg = layers.Dense(units=1024, activation="relu")(Rdx)
		return Model(inputs, rg,name="3dmhl")


	def MLP(self,pretrained_model):
                
		new_DL=pretrained_model.output
		new_DL=layers.Flatten()(new_DL)
		new_DL=layers.Dense(1024, activation="relu")(new_DL)   #64
		new_DL=layers.Dropout(self.do)(new_DL)
		new_DL=layers.Dense(512, activation="relu")(new_DL)    #64
		new_DL=layers.Dropout(self.do)(new_DL)
		new_DL=layers.Dense(self.classes, activation="sigmoid")(new_DL) #2
		return Model(inputs=pretrained_model.input, outputs=new_DL)
