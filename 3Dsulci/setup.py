from setuptools import setup
from setuptools import find_packages
#import pip

#conda install numba cffi -c drtodd13 -c conda-forge --override-channels
#pip.main(['install', 'git+https://www.github.com/keras-team/keras-contrib.git'])
#pip install git+https://github.com/JihongJu/keras-resnet3d.git
setup(name='s3D',
      version='0.2',
      description='Deep Learning 3D classification pipeline analysis in Python',
      url='',
      author='Michail Mamalakis',
      author_email='mm2703@cam.ac.uk',
      license='GPL-3.0+',
      packages=['s3D'],
      install_requires=[          
          'scikit-learn',
	  'torch',
          'vit-pytorch',
          'tensorflow==2.6.2',          #for the explainer petrubation need >=2.6.2 
          'tensorboard>=2.2.0',
          'fastprogress>=1.0.0',
	  'keras>=2.4.1',
          'seaborn',
          'quantus',
	  'six',
	  'einops>=0.3.0',
	  'h5py>=2.10.0',
          'numpy>=1.15.4',
          'scipy>=1.1.0',
	  'matplotlib>=3.1.0',
	  'dicom',
	  'pydicom>=2.1.1',	
	  'opencv-python>=4.0.0.21',
	  'Pillow>=5.3.0',
	  'vtk>=8.1.1',	
          'future',
	  'rq-scheduler>=0.7.0',
	  'med2image',
	  'imageio',
          'gensim',
          'SimpleItk',
          'networkx',
	  'pypac',
	  'nibabel',
	  'np_utils',
	  'medpy',
	  'onednn-cpu-gomp==2022.0.1',
	  'scikit-image',
	  'nets',
	  'shap',
      ],
	zip_safe=False

)


