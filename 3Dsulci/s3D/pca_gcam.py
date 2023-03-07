import skimage
import skimage.io
import skimage.transform
import numpy as np

import shap
import os
import csv

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.cm as cm
from s3D import create_3Dnet, quart
from tensorflow.keras.models import model_from_json
#from tensorflow import placeholder
import tensorflow.compat.v1 as tf1
#import tensorflow as tf1
#tf1.disable_v2_behavior()
from tensorflow.keras import backend as K

import tensorflow as tf
from skimage import io
from scipy import ndimage,misc
from skimage.transform import resize
from tensorflow.python.framework import ops
from scipy.stats import gmean
from tensorflow.python.ops import gen_nn_ops
#tf1.compat.v1.disable_eager_execution()
import glob
import cv2
import nibabel as nib
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
import skimage.transform as skTrans
from s3D import test
from s3D.test import *

# synset = [l.strip() for l in open('synset.txt').readlines()]
def image_preprocess(resized_inputs,channel=3):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    channel_means = tf1.constant([123.68, 116.779, 103.939],
        dtype=tf1.float32, shape=[1, 1, 1, channel], name='img_mean')
    return resized_inputs - channel_means


# returns image of shape [224, 224, 3]
# [height, width, depth]
def normal_extract(image):
    total=image.shape[0]
    std_mean=[]
    both=np.zeros(2)
    print(image.shape)
    for o in range(total):
        com_im=image[o]
        com_im=np.reshape(com_im,[com_im.shape[0]*com_im.shape[1]*com_im.shape[2]])
        both[1]=np.std(com_im)
        both[0]=np.mean(com_im)
        print(both)
        std_mean.append([np.std(com_im),np.mean(com_im)])
    return np.array(std_mean)

def load_image(path,label,label_name,xo=224,yo=224,zo=96, c=1,normalize=False):
    # load image
    image_list=[]
    im_path=[]
    img=[]
    y=[]
    image_tot=[]
    ii=0
    print(label_name)
    for op in label: 
        im_path = sorted(glob.glob(path +label_name[ii]+"/*"))
       # print(label_name[ii])
        print(im_path)
        total=np.array(im_path).shape[0]
        first_image_list=(im_path[0])
        print(first_image_list,total,im_path[1])
        scanw=nib.load(first_image_list)
        img.append(scanw.get_fdata())
        #img1=np.reshape(img1,[1,img1.shape[0],img1.shape[1],img1.shape[2]])
        for i in range(1,total):
            image_list=im_path[i]
            typei=(image_list)[-3:]
            if typei=='jpg':
                img.append(skimage.io.imread(im_path[i]))
            else:
                scan=nib.load(im_path[i])
                scan=scan.get_fdata()
                #scan=np.reshape(scan,[scan.shape[0],scan.shape[1],scan.shape[2]])
                img.append(scan)#,img,axis=0)
        img_a=np.array(img)
        total2=img_a.shape[0]
        total_img=[]
        for o in range(0,total2):
            if normalize:
                img1 = (img_a[o] / (np.max(img_a[o])))
                assert (0 <= img1).all() and (img1 <= 1.0).all()
            else:
                img1=img_a[o]
            resized_img=skTrans.resize(img1, (xo,yo,zo,1), order=1, preserve_range=True)
            total_img.append(resized_img)
        total_img=np.array(total_img)
        y_p=np.full(total_img.shape[0],op)
        y_p=np.reshape(y_p,[total_img.shape[0],1])
        print(y_p.shape,y_p[0])
        if ii==0:
            y=y_p
        else:
            y=np.append(y_p,y,axis=0)

        for t in range(y_p.shape[0]):
            image_tot.append(total_img[t])
        
        ii=ii+1
    
    img_t=np.array(image_tot)
    y=np.array(y)
    print(img_t.shape,y.shape)
    return img_t,y

def load_image2(path, normalize=False,xo=224,yo=224,zo=96,c=1):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    image_list=[]
    im_path=[]
    img=[]
    im_path = sorted(glob.glob(path +"/*"))
    total=np.array(im_path).shape[0]
    first_image_list=(im_path[0])
    print(first_image_list,total,im_path[2])
    scanw=nib.load(first_image_list)
    img.append(scanw.get_fdata())
    #img1=np.reshape(img1,[1,img1.shape[0],img1.shape[1],img1.shape[2]])
    for i in range(1,total):
        image_list=im_path[i]
        typei=(image_list)[-3:]
        if typei=='jpg': 
            img.append(skimage.io.imread(im_path[i]))
        else:
            scan=nib.load(im_path[i])
            scan=scan.get_fdata()
            #scan=np.reshape(scan,[scan.shape[0],scan.shape[1],scan.shape[2]])
            img.append(scan)#,img,axis=0)
    img_a=np.array(img)
    lenp=img_a.shape[0]
    y=np.ones([lenp,1])
    print(y.shape,img_a.shape)
    total2=img.shape[0]
    total_img=[]
    for o in range(0,total2):
        if normalize:
            img1 = (img_a[o] / (np.max(img_a[o])))
            assert (0 <= img1).all() and (img1 <= 1.0).all()
        else:
            img1=img_a[o]
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    #    short_edge = min(img1.shape[:2])
    #    yy = int((img1.shape[0] - short_edge) / 2)
    #    xx = int((img1.shape[1] - short_edge) / 2_b,)
     #   zz = int((img1.shape[2] - short_edge) / 2)
    #    crop_img = img1[ xx: xx + short_edge,yy: yy + short_edge, zz: zz + short_edge, :]
        # resize to 224, 224
        #print(np.array(crop_img).shape)
    #    for sz in range(img.shape[3]):
    #      resiz_xy = skimage.transform.resize(img1[:,:,sz], (1,xo,yo,1,c), preserve_range=True) # do not normalize at transform. 
    #        if sz==0:
    #            resized_xy=resiz_xy
    #        else:
    #            resized_xy=np.append(resized_xy,resiz_xy,axis=3)
    #    resized_xy=np.array(resized_xy)
    #    print(resized_xy.shape)
    #    resized_img= skimage.transform.resize(img1, (1,xo,yo,zo,c))#, preserve_range=True) # do not normalize at transform.
        resized_img=skTrans.resize(img1, (xo,yo,zo,1), order=1, preserve_range=True)
        total_img.append(resized_img)
    img_t=np.array(total_img)
    print(img_t[1,120,100,90,0],img_t[1,120,100,70,0],img_t[1,120,100,0,0])   
    return img_t

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1



def visualize2(image, conv_output, conv_grad, gb_vizi,store,x=224,y=224,z=96,num_chan=1,p=0):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad         
    img=255*image
    total=np.array(img).shape[0]
    #print("grad_val:",grads_val.shape)
    if (len(grads_val.shape)==5):
        weights = np.mean(grads_val, axis = (1, 2, 3)) # alpha_k, [512]
        c = np.zeros([output.shape[0],output.shape[1],output.shape[2], output.shape[3]], dtype = np.float32) # [7,7]
    else:
        weights = (grads_val) # alpha_k, [512]
        c = np.zeros([output.shape[0],output.shape[1],output.shape[2]], dtype = np.float32) # [7,7]
    #print("weights: ", weights.shape)
    #print("output: ",output.shape)
    #print("cam: ",c.shape)   
    cam=[]
    # Taking a weighted average
    for o in range(total):
        cam1=c[o]
        w_o=weights[o,:]
        for i, w in enumerate(w_o):
            if (len(grads_val.shape)==5):
                cam1 += w * output[o,:, :, :,i]
            else:
                cam1 += w * output[o, i]
            #print("w is : ",w)
            #print("output ; ", output[o,:5,:5,i], i)
        cam1=np.abs(cam1)
        camax = np.maximum(cam1, 0)
        camt = camax / np.max(camax)
        cam.append(camt)
     #   print("MAX cam is: ", np.max(camax), np.max(camt))

    cam=np.array(cam)#, dtype = np.uint8)
    #print("cam shape: ", cam.shape)
    cam = resize(cam, (total,x,y,z))#, preserve_range=True)
    img=resize(img,(total,x,y,z,num_chan))
    gb_vizi=resize(gb_vizi,(total,x,y,z))
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * c)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_ar=[]
    gb1_ar=[]
    im_ar=[]
    ch_at=[]
    # Create an image with RGB colorized heatmap
    for h in range(total):
    # Superimpose the heatmap on original image
      #  print(img[h,20,20,40,0],img[h,20,20,0,0],img[h,20,20,10,0])
        for j in range(img.shape[3]):
            im=np.array(img[h,:,:,j,:])
            ca=255*cam[h,:,:,j]
            im2d=tf1.keras.preprocessing.image.array_to_img(im)
            im2d=np.array(im2d)
            im2d=np.reshape(im2d,[1,img.shape[1],img.shape[2],1,img.shape[4]])
            cam_o2d=(ca).astype(np.uint8)
            ch1 = cv2.applyColorMap(cam_o2d, cv2.COLORMAP_JET)
            ch1= cv2.cvtColor(ch1, cv2.COLOR_BGR2RGB)
            ch1=np.reshape(ch1,[1,cam.shape[1],cam.shape[2],1,3])
            if j==0:
                im_a=im2d
                ch=ch1
            else:
                im_a=np.append(im_a,im2d,axis=3)
                ch=np.append(ch,ch1,axis=3)

        cam_o=(255*cam[h]).astype(np.uint8)
        cam_o=np.reshape(cam_o,[cam.shape[1],cam.shape[2],cam.shape[3],1])
        s_imq = np.array(cam_o) * 0.4 + np.array(img[h])
        s_im=(s_imq).astype(np.uint8)
        #s_im = cv2.applyColorMap(s_imq, cv2.COLORMAP_JET)
        jet_ar.append(s_im)  #tf1.keras.preprocessing.image.array_to_img(s_im)
        ch_at.append(ch)
        im_ar.append(im_a)
    s_imgh = jet_ar
    img2=im_ar
    ch2=ch_at
    #print("Max cam : ",np.max(ch2),ch2[10])
    #s_img=tf1.keras.preprocessing.image.array_to_img(s_imgh)
    s_img=s_imgh

    gb_viz=gb_vizi
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
    #gd_gb=cam
    #print("gp shape: ",np.array(gb_vizi).shape)

    if (len(gb_viz.shape)==5) and (gb_viz.shape[4]==3):
        gd_gb = np.dstack((
           gb_viz[:,:,:,:,0]* cam ,
            gb_viz[:,:, :,:, 1] * cam,
           gb_viz[:,:, :, :,2] * cam,
        ))
        gd_gb=[]
        gd_gb.append(gb_viz[:,:,:,:,0]*cam)
        gd_gb.append(gb_viz[:,:,:,:,1]*cam)
        gd_gb.append(gb_viz[:,:,:,:,2]*cam)
    #gb0=np.reshape(gd_gb,[total,gd_gb.shape[1],gd_gb.shape[1]],3)
    #img1=np.reshape(img,[total,img.shape[1],img.shape[1],num_chan])
    #gb1=np.array(gb0)*0.8 +np.array(img[h])
    #gd_gb== 255 * gd_gb / np.max(gd_gb)
    #gbw1=np.reshape(gb1,[gd_gb.shape[1],gd_gb.shape[2],gd_gb.shape[0]])
        gb0=np.array(gd_gb)
        gb0=np.reshape(gb0,[gb0.shape[1],gb0.shape[2],gb0.shape[3],gb0.shape[4],3])
    #for h in range(total):
     #      gb11=(255*gb0[h])*0.7 +np.array(img[h])
      #     gb_ar=tf1.keras.preprocessing.image.array_to_img(gb11)
       #    gb1_ar.append(gb_ar)
    #gb1=gb1_ar
    else:
        camadd=np.reshape(cam,[cam.shape[0],cam.shape[1],cam.shape[2],cam.shape[3],1])
        gb0=np.array(gb_viz* camadd)
    d1=np.array(ch2)
    d2=np.array(gb0)
    x1,y1,z1=d1.shape[2],d1.shape[3],d1.shape[4]
    x2,y2,z2=d2.shape[2],d2.shape[3],d2.shape[4]
    for i in range(total): 

        #   fig = plt.figure()    
         #  ax = fig.add_subplot(131, projection='3d')

           data=np.reshape(img2[i],[x,y,z])
         #  vox=ax.voxels(data, facecolors=plt.cm.gist_yarg(data),edgecolors=plt.cm.coolwarm(data))
         #  ax.set_title('Input Image')
         #  ax = fig.add_subplot(132, projection='3d')
           print(d1.shape,d2.shape)
           data1=255*np.reshape(d1[i,:,:,:,:,0],[x1,y1,z1])
           data11=255*d1[i,0,:,:,:,:]
         #  ax.voxels(data1, facecolors=plt.cm.coolwarm(data1),edgecolors=plt.cm.coolwarm(data1) )
         #  ax.set_title('Grad-CAM')   
           
         #  ax = fig.add_subplot(133, projection='3d')
           data2=np.reshape(d2[i,:,:,:,0],[x1,y1,z1])
         #  vox=ax.voxels(data2, facecolors=plt.cm.coolwarm(data2), edgecolors=plt.cm.coolwarm(data2))
         #  ax.set_title('guided GCAM')
         #  plt.savefig(store+'/image_'+str(i)+str(p)+'.png')
           
           imgnthree=nib.Nifti1Image(data, affine=np.eye(4))
           str4=store + '/case_%s_fig_%s.%s' % (p,'img','nii.gz')
           imgnthree.header.set_data_dtype(np.uint8)
           nib.save(imgnthree,str4)
           imgnthree1=nib.Nifti1Image(data11, affine=np.eye(4))
           str41=store + '/case_%s_fig_%s.%s' % (p,'GC','nii.gz')
           imgnthree1.header.set_data_dtype(np.uint8)
           nib.save(imgnthree1,str41)

           imgnthree2=nib.Nifti1Image(data2, affine=np.eye(4))
           str42=store + '/case_%s_fig_%s.%s' % (p,'GGC','nii.gz')
           imgnthree2.header.set_data_dtype(np.uint8)
           nib.save(imgnthree2,str42)
           
          # plt.close(fig)
           
           
    return np.array(ch2)


@ops.RegisterGradient("GuidedRelu2")
def _GuidedReluGrad(op, grad):
    return tf1.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf1.zeros_like(grad))


#from slim.nets import resnet_v1
#slim = tf.contrib.slim

def layers_print2(model,inputs):
     activations=[]
     inp = model.input                              # input placeholder
     outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
     functors = [K.function(inp, [out]) for out in outputs]   # evaluation functions
     # Testin
     list_inputs = inputs
    # layer_outputs = [func(list_inputs)[0] for func in functors]
     #for layer_activations in layer_outputs:
      #   activations.append(layer_activations)

     return outputs,functors  #,activations



def viewer(X,model,store, point=[7,9],model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='none',input_n=1,casemean='mean',start=0,end=0):
    tf1.disable_v2_behavior()
    #input one class per time to create the pca map
    # all the images same class in other case wrong
    tf1.compat.v1.disable_eager_execution()
    img1 = load_image2(X,xo=height,yo=width,zo=depth, c=channels) #"./demo.png", normalize=False)
    #img2=load_image2(X,normalize=True,xo=height,yo=width,c=channels,data_aug=True)
    #im_t=normal_extract(img1)
    batch_size=np.array(img1).shape[0]
    #img1=ndimage.rotate(img1,rot,reshape=False)
    print(img1.shape)
    blabel=[]
    batch1_img_all = np.array(img1)
    batch1_img=batch1_img_all
    #batch2_img=batch2_img_all
    #batch_size=1
    label_binary=label
    batch1_labe = np.array(label,dtype=np.float32)  # 1-hot result for Boxer
    batch1_labe = batch1_labe.reshape(1, -1)
    for o in range(batch_size):
        blabel.append(batch1_labe)
    batch1_label=np.array(blabel)
    imageK=K.placeholder(shape=(batch_size,height,width,depth,channels))
    label = K.placeholder(shape=(batch_size, classes))
    #if input_n>=2:
    #    batch1_label=np.array(blabel)
    #    while batch1_img.shape[3]<(input_n):
    #        batch1_img=np.concatenate((batch1_img,batch1_img_all),axis=3)

    print("The model name is: ",model_name)
    cn=create_3Dnet.create_3Dnet(model_name,height,width,depth,channels,classes,name="",do=0.3)
    model_structure=cn.model_builder()
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    model_str=model_structure
    model_str.load_weights(file_store)
    end_p ,functors= layers_print2(model_str,batch1_img)
    end_points=end_p
    print(len(end_points))
    deliver=end_p
    prob =(deliver[(len(end_points)-point[0])])  #,dtype=tf1.float32) # after softmax
    cost = (-1) * tf1.reduce_sum(tf1.multiply(batch1_label, tf1.log(prob)), axis=1) #one image per time in other case need modification 
    target_conv_layer =deliver[len(end_points)-point[1]]
    net=model_str.output
    y_c = tf1.reduce_sum(tf1.multiply(net, batch1_label), axis=1)#net
    target_conv_layer_grad = K.gradients(y_c, target_conv_layer)[0]

    # Guided backpropagtion back to input layer
    gb_grad = K.gradients(cost[1], model_str.input)[0] #[0] before
    iterate=K.function([model_str.input,label],[cost,gb_grad,prob])
    iterate2=K.function([model_str.input,label],[y_c,target_conv_layer_grad,target_conv_layer])
    y_t=batch1_label[0]
    im_total=[]
    end_=batch_size
    if end!=0:
        end_=end
    for i in range(start,end_):
        bi=np.reshape(batch1_img[i],[1,batch1_img.shape[1],batch1_img.shape[2],batch1_img.shape[3],batch1_img.shape[4]])
        bl=np.reshape(batch1_label[i],[1,batch1_label.shape[1],batch1_label.shape[2]])
        cost_np,gb_grad_value, prob_np=iterate([bi,bl])
        print(bi.shape,batch_size)
        y_c_np,target_conv_layer_grad_value,tg=iterate2([bi,bl])#([tg,batch1_img])
        target_conv_layer_value=tg
    #print("prob after: ", prob_np)
    #print("prob image_active: ", prob_im)
    #print("Y_c ",y_c)     
    #print("Y_C result= ",y_c_np)
        y_pred = model_str.predict(bi)
   # print("prediction= ",y_pred)  
    #print("CCN  grad out: ",target_conv_layer_grad_value)
    #print("CNN before: ", target_conv_layer)
    #print("CNN after: ",target_conv_layer_value)
    #print("CNN image: ",tg)
        y_pred_str=map(str,y_pred)
        y_t=np.append(y_pred,y_t,axis=0)

        imagin=visualize2(bi, target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value,store,x=height,y=width,z=depth,num_chan=channels,p=i)
        imagin=np.reshape(imagin,[imagin.shape[1],imagin.shape[2],imagin.shape[3],imagin.shape[4],imagin.shape[5]])
        print('saved ',i, imagin.shape)
        if i==start:
            im_total=imagin
        else:
            im_total=np.append(im_total,imagin,axis=0)

    im_total=np.array(im_total)
    print(im_total.shape)
    if len(im_total.shape)==6:
            im_total=np.reshape(im_total,[im_total.shape[0]*im_total.shape[1],im_total.shape[2],im_total.shape[3],im_total.shape[4],im_total.shape[5]])
    dim=4
    print(batch_size)
    if batch_size>=18:
 
            dim=6
    else:
            dim=batch_size
    mean(im_total,batch1_img,store,casemean,dim)




    string_ints = [str(inta) for inta in label_binary]
    label_str = "_".join(string_ints)
    ns="/y_pred_"+"%s_%s.csv" %(model_name,label_str)
    ns2="/STD_MEAN_"+"%s.csv" %(label_str)
    store2=store

    with open(store2+ns, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        y_pred_str=zip([map(str,y_t[:,u]) for u in range((np.array(y_t).shape[1]))])
        print("y_pred prediction:")
        for row in y_pred_str:
            for t in row:
                employee_writer.writerow(t)
                print(t)
    im_t=im_total
    with open(store2+ns2, mode='w') as employee_file2:
        employee_writer2 = csv.writer(employee_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        img_std_m=zip([map(str,im_t[:,u]) for u in range((np.array(im_t).shape[1]))])
        print(im_t.shape)
        print("STD and Mean values of input image:")
        for row2 in img_std_m:
            for t2 in row2:
                employee_writer2.writerow(t2)
                print(t2)
                
                
def mean(X,Y,store,case='mean',dim=1):
    print("image shape is: ",Y.shape)
##############################################################################
    #filter the peripheral organs patches 
    X=np.array(X/255)
    Y=np.array(Y/255)
    r=(Y)
    g=(X)
    r=np.array(r)
    g=np.array(g)
    print(r.shape)
############################################################################
 
    st='Average'
    if case=='mean':
        imn2=gaverage(r)
        imn=gaverage(g)
    elif case=='tsne':
        imn2,st=t_sne(r)
        imn,st=t_sne(g)
    elif case=='pca':
        imn2,st=pca(r,com=dim, c='shape',store=store)
        imn,st=pca(g,com=dim, c="GradCam",store=store)
        st1=std(r)
        st2=std(g)
        print(np.array(st1).shape)
    elif case=='lda':
        imn2,st=lda(r.dim)
        imn,st=lda(g,dim)
    im2=(255*imn2).astype(np.uint8)
    im=(255*imn).astype(np.uint8)
    im3=(255*np.array(st1)).astype(np.uint8) 
    im4=(255*np.array(st2)).astype(np.uint8)
    im3=np.reshape(im3,[im3.shape[1],im3.shape[2],im3.shape[3],im3.shape[4]])
    im4=np.reshape(im4,[im4.shape[1],im4.shape[2],im4.shape[3],im4.shape[4]])
    print(im3.shape)
    for j in range(im3.shape[3]):
        im1=np.array(im[:,:,j,:])   
        im21=np.array(im2[:,:,j,:])
        im31=np.array(im3[:,:,j,:])
        im41=np.array(im4[:,:,j,:])
        im10 = cv2.applyColorMap(im1, cv2.COLORMAP_JET)
        im20= cv2.applyColorMap(im21, cv2.COLORMAP_JET)
        im30 = cv2.applyColorMap(im31, cv2.COLORMAP_JET)
        im40 = cv2.applyColorMap(im41, cv2.COLORMAP_JET)

        if j==0:
            imt1=im10
            imt2=im20
            imt3=im30
            imt4=im40
        else:
            imt1=np.append(imt1,im10,axis=3)
            imt2=np.append(imt2,im20,axis=3)
            imt3=np.append(imt3,im30,axis=3)
            imt4=np.append(imt4,im40,axis=3)
            
    plt.figure()
    fig = plt.figure()
    print(imt2.shape)
    ax = fig.add_subplot(141, projection='3d')
    data4=np.reshape(imt2,[imt2.shape[0],imt2.shape[1],imt2.shape[2]])
    vox=ax.voxels(data4, facecolors=plt.cm.gist_yarg(data4),edgecolors=plt.cm.coolwarm(data4))
    ax.set_title('Average Image ')
    ax = fig.add_subplot(142, projection='3d')
    data5=np.reshape(imt1,[imt1.shape[0],imt1.shape[1],imt1.shape[2]])
    vox=ax.voxels(data5, facecolors=plt.cm.gist_yarg(data5),edgecolors=plt.cm.coolwarm(data5))
    ax.set_title(st+' GRANCAM ')

    ax = fig.add_subplot(143, projection='3d')
    data6=np.reshape(imt3,[imt3.shape[0],imt3.shape[1],imt3.shape[2]])
    vox=ax.voxels(data6, facecolors=plt.cm.gist_yarg(data6),edgecolors=plt.cm.coolwarm(data6))
    ax.set_title('std Average Image ')
    ax = fig.add_subplot(144, projection='3d')
    data7=np.reshape(imt4,[imt4.shape[0],imt4.shape[1],imt4.shape[2]])
    vox=ax.voxels(data7, facecolors=plt.cm.gist_yarg(data7),edgecolors=plt.cm.coolwarm(data7))
    ax.set_title(st+' std GRANCAM ')
    plt.savefig(store+'/image_FE_'+st+'_Gmean.png')
    plt.close(fig)


def gaverage(X):
    lenf=X.shape[3]
    aver=np.zeros([1,X.shape[1],X.shape[2],X.shape[3],X.shape[4]])
    for o in range (X.shape[0]):
        aver=aver+X[o]/lenf
    average=np.reshape(aver,[X.shape[1],X.shape[2],X.shape[3],X.shape[4]])
    return average    

def std(X):
    stdv=np.std(X, axis=0, keepdims=True)
    return stdv



def pca(X,com=18,c='none',store='none'):
    dim=X.shape[0]
    feat=X.shape[1]*X.shape[2]*X.shape[3]*X.shape[4]
    ro=np.reshape(X,[dim,feat])
    pca_r = PCA(n_components=com)
    pca_r_trans= pca_r.fit_transform(ro)
    pca_i=pca_r.inverse_transform(pca_r_trans)
    im2=np.reshape(pca_i,[dim,X.shape[1],X.shape[2],X.shape[3],X.shape[4]])
    imo=np.reshape(im2[0],[X.shape[1],X.shape[2],X.shape[3],X.shape[4]])
    for o in range(com):
        im1=im2[o,:,:,:,:] #0 median here was 0 not all : in the channels
        imin=np.min(im1)
        imax=np.max(im1)
        print(np.max(im1),np.min(im1))
        maxt=np.max([np.abs(imin),np.abs(imax)])
        dataplot=255*(im1/maxt) #normalize everything
        print(np.max(dataplot),np.min(dataplot))
                
        dataplot=np.reshape(dataplot,[X.shape[1],X.shape[2],X.shape[3],X.shape[4]])
        imgnthr=nib.Nifti1Image(dataplot, affine=np.eye(4))
        strq=store + '/PCA_%s_fig_%s.%s' % (o,c,'nii.gz')
        imgnthr.header.set_data_dtype(np.float)
        nib.save(imgnthr,strq)
        
    typr='PCA'
    return imo, typr

def t_sne(X,com=1):
    r=X    
    dim=r.shape[0]
    feat=r.shape[1]*r.shape[2]*r.shape[3]*r.shape[4]
    X=np.reshape(X,[feat,dim])
    Xe = TSNE(n_components=com, learning_rate='auto', init='random')
    X_s=Xe.fit_transform(X)
    #o=Xe.inverse_transform(X_s)
    im=np.reshape(X_s,[com,r.shape[1],r.shape[2],r.shape[3],r.shape[4]])
    imo=np.reshape(im[0],[r.shape[1],r.shape[2],r.shape[3],r.shape[4]])
    typr='t_SNE'
    return imo, typr

def lda(X,com=2):
    r=X
    dim=r.shape[0]
    feat=r.shape[1]*r.shape[2]*r.shape[3]*r.shape[4]
    X=np.reshape(X,[feat,dim])
    lda=LatentDirichletAllocation(n_components=com)
    lda.fit(X)
    o=lda.transform(X)
    c=lda.inverse_transform(o)
    im=np.reshape(c,[com,r.shape[1],r.shape[2],r.shape[3],r.shape[4]])
    imo=np.reshape(im[0],[r.shape[1],r.shape[2],r.shape[3],r.shape[4]])
    typr='LDA'
    return imo, typr


# plot actual and predicted class
def plot_actual_predicted(images, pred_classes,store,number_class=2):
    fig, axes = plt.subplots(1, number_class, figsize=(16, 15))
    axes = axes.flatten()
    # plot
    ax = axes[0]
    dummy_array = np.array([[[0, 0, 0, 0]]], dtype='uint8')
    ax.set_title("Base reference")
    ax.set_axis_off()
    ax.imshow(dummy_array, interpolation='nearest')
    # plot image
    for k,v in images.items():
        ax = axes[k+1]
        ax.imshow(v, cmap=plt.cm.binary)
        ax.set_title(f"True: %s \nPredict: %s" % (class_names[k], class_names[pred_classes[k]]))
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(store+'/image_shap_total.png')
    plt.close(fig)


def shap_m(model,x_test,store,class_names=['none']):
    #https://stackoverflow.com/questions/52137579/save-shap-summary-plot-as-pdf-svg
     #https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
    x_test=np.array(x_test)
    one=x_test[0]
    zero=np.zeros(one.shape)
    #background = x_test[np.random.choice(x_test.shape[0], 1, replace=False)]
    zero=np.reshape(zero,[1,zero.shape[0],zero.shape[1],zero.shape[2],zero.shape[3]])
    explainer = shap.DeepExplainer((model.layers[0].input,model.layers[-1].output), zero)
    # compute shap values
    o=0
    shapv=[]
    for o in range(x_test.shape[0]):
        plt.clf()
        x_test_each=np.reshape(x_test[o],[1,x_test.shape[1],x_test.shape[2],x_test.shape[3],x_test.shape[4]])
        x_test_each=x_test_each
        shap_value=explainer.shap_values(x_test_each,check_additivity=False)
        shap_value=np.array(shap_value)
        shap_value=np.reshape(np.array(shap_value),[1,shap_value.shape[0],shap_value.shape[2],shap_value.shape[3],shap_value.shape[4],shap_value.shape[5]])
        shapv.append(shap_value)
        shapva=np.array(shapv)   
        #print(np.max(shapva),np.min(shapva))
    return shapva

def shap_preprocessing(X,model,store,model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='none',class_names=['pcs','n_pcs']):
    #tf1.disable_v2_behavior()
    tf1.compat.v1.disable_eager_execution() 
    x_test,y_test= load_image(X,label,class_names,xo=height,yo=width,zo=depth, c=channels)
    #x_test_dict = dict()
    #print(y_test.shape)
    #print(x_test_dict.keys())
    y_test.reshape(y_test.shape[0],)
    #for i, l in enumerate(y_test):
    #    if len(x_test_dict)==len(label):
    #        break
    #    if l not in x_test_dict.keys():
    #        x_test_dict[l] = x_test[i]
    #x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
    #x_test_each_class = np.asarray(x_test_each_class)

    print("The model name is: ",model_name)
    cn=create_3Dnet.create_3Dnet(model_name,height,width,depth,channels,classes,name="",do=0.3)
    model_structure=cn.model_builder()
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    model_str=model_structure
    model_str.load_weights(file_store)

    #predictions = model_str.predict(x_test_each_class)
    #predicted_class = np.argmax(predictions, axis=1)

    # select backgroud for shap
    up_b=350
    dim=8
    np.random.shuffle(x_test)
    x_t=x_test[:up_b]
    #data_for_pred=x_test_each_class #.iloc[2,:].astype(float)
    shaps=shap_m(model_str,x_t,store,class_names) 
    shaps=np.reshape(shaps,[shaps.shape[0],shaps.shape[1]*shaps.shape[2],shaps.shape[3],shaps.shape[4],shaps.shape[5],shaps.shape[6]])
    print('The shape values of shap is: ',shaps.shape)

    mean(shaps[:,0,:,:,:,:],x_t,(store+'/PCA'),'pca',dim)
    print('store the noPCA..')
    mean(shaps[:,1,:,:,:,:],x_t,(store+'/noPCA'),'pca',dim)


def petrubation_m(X,Xp,portion=0.75,shap_v='off'):
    threshold_up=np.max(Xp)*portion
    threshold_down=np.min(Xp)*portion
    Xpetrubation=[]
    for i in range(X.shape[0]):
        xpetrubation=X[i]
        xp=Xp[i]
        if shap_v=='off':
            numbers=np.where(xp>=threshold_up)
            xpetrubation[numbers]=0
           

        else:
            length=np.where(xp[0]>=threshold_up)
            print(len(length),xp[0].shape,threshold_up)
            print(np.array(length).shape)
            for o in range (1,xp.shape[0]):
                numbers=np.where(xp[o]>=threshold_up)
                if (np.array(numbers).shape[1]>np.array(length).shape[1]):
                    print(np.array(numbers).shape)
                    length=numbers
            xpetrubation[length]=0

            #xpm=np.where(np.min([np.min(xp[o]) for o in range (xp.shape[0])])
            #numbers2=[np.where(xp[xpm]<=threshold_down)]
            #xpetrubation[numbers2]=np.max(x)

        Xpetrubation.append(xpetrubation)
    Xpetrubation=np.array(Xpetrubation)
    return Xpetrubation

def petrubation_shap(X,model,store,petrubation='mine',model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='simple_3d',class_names=['pcs','n_pcs'],ratio=0.5):
    tf1.disable_v2_behavior()
    tf1.compat.v1.disable_eager_execution()
    x_test,y_test= load_image(X,label,class_names,xo=height,yo=width,zo=depth, c=channels)
    x_test_dict = dict()
    print(y_test.shape)
    print(x_test_dict.keys())
     #y_test.reshape(y_test.shape[0],)
    
    print("The model name is: ",model_name)
    cn=create_3Dnet.create_3Dnet(model_name,height,width,depth,channels,classes,name="",do=0.3,backbone=backbone)
    model_structure=cn.model_builder()
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    model_str=model_structure
    model_str.load_weights(file_store)

    # select backgroud for shap
    up_b=8
    rng_state = np.random.get_state()
    np.random.shuffle(x_test)
    x_t=x_test[:up_b]
    shaps=shap_m(model_str,x_t,store,class_names)
    shaps=np.reshape(shaps,[shaps.shape[0],shaps.shape[1]*shaps.shape[2],shaps.shape[3],shaps.shape[4],shaps.shape[5],shaps.shape[6]])

    np.random.set_state(rng_state)
    np.random.shuffle(y_test)
    labels1=y_test[:up_b]

    if petrubation=='mine':
        new_data=petrubation_m(x_t,shaps,ratio,'on')
        test_result(new_data,labels1,model_str, model_name,store,class_names)
        test_result(x_t,labels1,model_str, model_name,store,class_names) 
    else:
        for i in range(shaps.shape[1]):
            store_new=(store+str(i)+'/')
            if not os.path.exists(store_new): os.makedirs(store_new)
            quart.quart(store_new, model_str,x_t,labels1,shaps[:,i,:,:,:,:],Faithfulness='on',Robustness='off',Complexity='off')

def petrubation_grad(X,model,store,petrubation='mine',point=[7,9],model_name='none',height=256,width=256,depth=96,channels=1,classes=1,case='test',batch=1,label=[0,1],rot=0,backbone='simple_3d',class_names=['pcs','n_pcs'],ratio=0.5):

    tf1.disable_v2_behavior()
    #tf1.compat.v1.disable_eager_execution()
    x_test,y_test= load_image(X,label,class_names,xo=height,yo=width,zo=depth, c=channels)
    print(y_test.shape)
    print("The model name is: ",model_name)
    cn=create_3Dnet.create_3Dnet(model_name,height,width,depth,channels,classes,name="",do=0.3,backbone=backbone)
    model_structure=cn.model_builder()
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    model_str=model_structure
    model_str.load_weights(file_store)
    # select 
    up_b=8
    rng_state = np.random.get_state()
    np.random.shuffle(x_test)
    x_t=x_test[:up_b]
    np.random.set_state(rng_state)
    np.random.shuffle(y_test)
    labels1=y_test[:up_b]
    
    tf1.compat.v1.disable_eager_execution()
    batch_size=np.array(x_t).shape[0]
    blabel=[]
    batch1_img_all = np.array(x_t)
    batch1_img=batch1_img_all
    label_binary=labels1
    batch1_labe = np.array(labels1,dtype=np.float32)  # 1-hot result for Boxer
    #batch1_labe = batch1_labe.reshape(1, -1)
    #for o in range(batch_size):
    #    blabel.append(batch1_labe)
    #batch1_label=np.array(labels1)
    batch1_label=np.array(batch1_labe)
    #tf1.keras.backend as K
    imageK= K.placeholder(shape=(batch_size,height,width,depth,channels))
    label = K.placeholder(shape=(batch_size, classes))
    end_p ,functors= layers_print2(model_str,batch1_img)
    end_points=end_p
    print(len(end_points),batch1_img.shape,batch1_label.shape)
    deliver=end_p
    prob =(deliver[(len(end_points)-point[0])])  #,dtype=tf1.float32) # after softmax
    cost = (-1) * tf1.reduce_sum(tf1.multiply(batch1_label, tf1.log(prob)), axis=1) #one image per time in other case need modification 
    target_conv_layer =deliver[len(end_points)-point[1]]
    net=model_str.output
    y_c = tf1.reduce_sum(tf1.multiply(net, batch1_label), axis=1)#net
    target_conv_layer_grad = K.gradients(y_c, target_conv_layer)[0]

    # Guided backpropagtion back to input layer
    gb_grad = K.gradients(cost[1], model_str.input)[0] #[0] before
    iterate=K.function([model_str.input,label],[cost,gb_grad,prob])
    iterate2=K.function([model_str.input,label],[y_c,target_conv_layer_grad,target_conv_layer])
    y_t=batch1_label[0]
    im_total=[]
    
    for i in range(batch_size):
        bi=np.reshape(batch1_img[i],[1,batch1_img.shape[1],batch1_img.shape[2],batch1_img.shape[3],batch1_img.shape[4]])
        bl=np.reshape(batch1_label[i],[1,batch1_label.shape[1]])
        cost_np,gb_grad_value, prob_np=iterate([bi,bl])
        y_c_np,target_conv_layer_grad_value,tg=iterate2([bi,bl])#([tg,batch1_img])
        target_conv_layer_value=tg
        y_pred = model_str.predict(bi)
        y_pred_str=map(str,y_pred)
        imagin=visualize2(bi, target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value,store,x=height,y=width,z=depth,num_chan=channels,p=i)
        imagin=np.reshape(imagin,[imagin.shape[1],imagin.shape[2],imagin.shape[3],imagin.shape[4],imagin.shape[5]])
        print('saved ',i, imagin.shape)
        if i==0:
            im_total=imagin
            y_t=y_pred
        else:
            im_total=np.append(im_total,imagin,axis=0)
            y_t=np.append(y_pred,y_t,axis=0)
    im_total=np.array(im_total)
    if len(im_total.shape)==6:
            im_total=np.reshape(im_total,[im_total.shape[0]*im_total.shape[1],im_total.shape[2],im_total.shape[3],im_total.shape[4],im_total.shape[5]])
    im_total=rgb2gray(im_total)
    im_total=np.reshape(im_total,[im_total.shape[0],im_total.shape[1],im_total.shape[2],im_total.shape[3],1])
    print(im_total.shape,x_t.shape)
    print(im_total[0])
    #os.environ['TF2_BEHAVIOR'] = '1' 
    if petrubation=='mine':
        new_data=petrubation_m(x_t,im_total,ratio,'off')
        test_result(new_data,labels1,model_str, model_name,store,class_names)
        test_result(x_t,labels1,model_str, model_name,store,class_names)
    else:
        quart.quart(store,model_str,x_t,labels1,im_total,Complexity='on',Faithfulness='on',Robustness='on')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
