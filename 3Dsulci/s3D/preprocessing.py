#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function

# name of weights store of main segmentation
import nibabel as nib
from scipy import ndimage

class preprocessing:


	def __init__(self,h,w,d,format_file='nii'):
		self.h=h
		self.w=w
		self.d=d
		self.format_file=format_file

	def read_nifti_file(self,filepath):
		"""	Read and load volume"""
    		# Read file
	
		if self.format_file=='nii':
			scan = nib.load(filepath)
		else:
			scan = nib.load(filepath('gifti', 'ascii.gii'))
    		# Get raw data
		scan = scan.get_fdata()
		return scan


	def normalize(self,volume):
		"""Normalize the volume"""
		min = -1000
		max = 400
		volume[volume < min] = min
		volume[volume > max] = max
		volume = (volume - min) / (max - min)
		volume = volume.astype("float32")
		return volume


	def resize_volume(self,img):
		"""Resize across z-axis"""
    		# Set the desired depth
		desired_depth = self.d
		desired_width = self.w
		desired_height = self.h
    		# Get current depth
		current_depth = img.shape[-1]
		current_width = img.shape[0]
		current_height = img.shape[1]
		# Compute depth factor
		depth = current_depth / desired_depth
		width = current_width / desired_width
		height = current_height / desired_height
		depth_factor = 1 / depth
		width_factor = 1 / width
		height_factor = 1 / height
		# Rotate
		img = ndimage.rotate(img, 90, reshape=False)
		# Resize across z-axis
		img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
		return img


	def process_scan(self,path):
		"""Read and resize volume"""
    		# Read scan
		volume = self.read_nifti_file(path)
    		# Normalize
		volume = self.normalize(volume)
		# Resize width, height and depth
		volume = self.resize_volume(volume)
		return volume


