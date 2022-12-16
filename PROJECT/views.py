from django.shortcuts import render
import datetime
from django.utils import timezone
from account.models import *
from PIL import Image
from .models import *
from django.conf import settings  
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2
import pywt
from django.db.models import Q
import pywt.data
import numpy as np
import math
from .forms import *
from PIL import Image, ImageFilter
from skimage import img_as_float
from skimage.util import random_noise
from skimage.io import imread, imsave
from PIL import Image
import matplotlib.image as mpimg
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
# libraries that are generally imported for any deep learning model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# django_project/users/views.py
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
# django_project/users/views.py
from django.contrib.auth.forms import AuthenticationForm

# libraries need to prepare the data
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# libraries required to build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# libraries for activation functions required
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import softmax

# weight initializer libraries
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.initializers import HeUniform

# optimizer library
from tensorflow.keras.optimizers import Adam

# callback library
from tensorflow.keras.callbacks import EarlyStopping
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required

from django.contrib.auth.forms import AuthenticationForm

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap("Spectral")
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy as sp
import pylab as pl
from skimage import io


"""
Package                      Version
---------------------------- ----------
absl-py                      1.3.0
asgiref                      3.5.2
astunparse                   1.6.3
cachetools                   5.2.0
certifi                      2022.9.24
charset-normalizer           2.1.1
contourpy                    1.0.6
cycler                       0.11.0
Django                       4.1.3
django-crispy-forms          1.14.0
flatbuffers                  22.10.26
fonttools                    4.38.0
gast                         0.4.0
google-auth                  2.14.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.50.0
h5py                         3.7.0
idna                         3.4
imageio                      2.22.3
keras                        2.10.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.4
libclang                     14.0.6
Markdown                     3.4.1
MarkupSafe                   2.1.1
matplotlib                   3.6.2
networkx                     2.8.8
numpy                        1.22.4
oauthlib                     3.2.2
opencv-python                4.6.0.66
opt-einsum                   3.3.0
packaging                    21.3
pandas                       1.5.1
Pillow                       9.3.0
pip                          22.2.2
protobuf                     3.19.6
pyasn1                       0.4.8
pyasn1-modules               0.2.8
pyparsing                    3.0.9
python-dateutil              2.8.2
pytz                         2022.6
PyWavelets                   1.4.1
requests                     2.28.1
requests-oauthlib            1.3.1
rsa                          4.9
scikit-image                 0.19.3
scipy                        1.7.3
setuptools                   65.4.1
six                          1.16.0
sqlparse                     0.4.3
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.10.0
tensorflow-estimator         2.10.0
tensorflow-io-gcs-filesystem 0.27.0
termcolor                    2.1.0
tifffile                     2022.10.10
typing_extensions            4.4.0
urllib3                      1.26.12
Werkzeug                     2.2.2
wheel                        0.37.1
wrapt                        1.14.1

"""

#-----------------------------redirection for home page---------------------------------------

def homepage(request):
    writing=WritingImage.objects.order_by('-writing').values()[:2]    
    print(writing)
    return render(request,'index.html')

#-----------------------------redirection for dashboard---------------------------------------

def dashboard(request):
    return render(request,'home/index.html')

#-----------------------------redirection for index page--------------------------------------

def index(request):
    im = Image.open("media/images/1.jpeg")
    # #Show rotated Image
    im = im.rotate(45)
    up=Denoise.objects.all().last()
    img=Denoise.objects.filter(caption=up).last()
    print(img)
    im.save("media/ouput/out.png")
    return render(request,'uploading.html',{"img":img, 'media_url':settings.MEDIA_URL})

#-----------------------------redirection for noise removal--------------------------------------
def noise_removal(request):
    
    pathstring=str
    up=Denoise.objects.last()
    print(up)
    img=Denoise.objects.filter(caption=up).last()
    print("img : ",img)
    pathstring="media/images/"+str(img)
    print(pathstring)
    img1 = []
    # get images from dataset and put it in list
    img1.extend([pathstring])
    print(img1)
    # save greyscale images in the selected path
    img2 = []
    img2.extend(["media/ouput/greyscale.JPG"])
    # save images with gaussian noise in the selected path
    img3 = []
    img3.extend(["media/ouput/guassiannoise.JPG"])
    # save images with median filter in the selected path
    img4 = []
    img4.extend(["media/ouput/medianfilter.JPG"])
    # save images with gaussian filter in the selected path
    img5 = []
    img5.extend(["media/ouput/gaussian filter.JPG"])
    # save images with unsharp filter in the selected path
    img6 = []
    img6.extend(["media/ouput/unsharp filter.JPG"])

    def psnr(img1, img2): 
        # count PSNR of the image with filter compared to original greyscale image
        mse = np.mean((img1-img2)**2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnr_noisy = [0 for x in range(len(img1))]
    psnr_median = [0 for x in range(len(img1))]
    psnr_gaussian = [0 for x in range(len(img1))]
    for i in range(len(img1)):
        img_original = cv2.imread(img1[i],cv2.IMREAD_UNCHANGED)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        #Greyscale Image
        img_greyscale = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(img2[i], img_greyscale)
        #Add Gaussian Noise
        img_greyscale = cv2.imread(img2[i])
        mean = 100
        var = 100
        sigma = var **0.8
        gaussian = np.random.normal(mean, sigma, (img_greyscale.shape[0],img_greyscale.shape[1])) #  np.zeros((224, 224), np.float32)
        img_noisy = np.zeros(img_greyscale.shape, np.float32)
        if len(img_greyscale.shape) == 2:
            img_noisy = img_greyscale + gaussian
        else:
            img_noisy[:, :, 0] = img_greyscale[:, :, 0] + gaussian
            img_noisy[:, :, 1] = img_greyscale[:, :, 1] + gaussian
            img_noisy[:, :, 2] = img_greyscale[:, :, 2] + gaussian
        cv2.normalize(img_noisy, img_noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        img_noisy = img_noisy.astype(np.uint8)
        cv2.imwrite( img3[i], img_noisy)
        #Use Median Filter
        im = Image.open(img3[i])
        img_median = im.filter(ImageFilter.MedianFilter)
        #Use Gaussian Filter
        img_gaussian = cv2.GaussianBlur(img_noisy, (5, 5),20)
        cv2.imwrite(img5[i], img_gaussian)
        #Count PSNR
        psnr_noisy[i] = psnr(img_greyscale, img_noisy)
        psnr_median[i] = psnr(img_greyscale, img_median)
        psnr_gaussian[i] = psnr(img_greyscale, img_gaussian)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 8))
        fig.text(0.1,0.12,
                'Hasil perhitungan PSNR image gaussian noise = %s\n'
                'Hasil perhitungan PSNR image dengan median filter = %s\n'
                'Hasil perhitungan PSNR image dengan gaussian filter = %s' 
                % (psnr_noisy[i],
                psnr_median[i],
                psnr_gaussian[i]),
                fontsize=13)
        ax1.imshow(img_greyscale)
        ax1.set_title("ORIGINAL")
        ax1.axis('off')
        ax2.imshow(img_noisy)
        ax2.set_title("GAUSSIAN NOISE")
        ax2.axis('off')
        print(img_median)
        img_median.save('media/ouput/img_median.JPG')
        ax3.imshow(img_median)
        ax3.set_title("MEDIAN FILTER")
        ax3.axis('off')
        ax4.imshow(img_gaussian)
        ax4.set_title("GAUSSIAN FILTER")
        ax4.axis('off')
    img.image_output="media/ouput/img_median.JPG"
    print("Rata-rata PSNR image gaussian noise =" , (sum(psnr_noisy)/ len(psnr_noisy)))
    print("Rata-rata PSNR image dengan median filter =" , (sum(psnr_median)/ len(psnr_median)))
    print("Rata-rata PSNR image gaussian filter =" , (sum(psnr_gaussian)/ len(psnr_gaussian)))
    pathstring=""
    return render(request,'home/Noise_display.html',{"img":img, 'media_url':settings.MEDIA_URL})

#-----------------------------redirection for image upload for noise removal---------------------------------------
# Create your views here.

def image_upload(request):
    if request.method == 'POST':
        form = Denoiseform(request.POST, request.FILES)
        uploaded_file=request.FILES
        if form.is_valid():
            model_instance = form.save(commit=False)
            model_instance.caption=uploaded_file['image'].name
            form.save()
            return redirect('/noise_removal')
    else:
        form = Denoiseform()
        return render(request, 'upload.html', {'form' : form})
    return render(request, 'upload.html', {'form' : form})
    
#--------------------------------------------classification image upload --------------------------------

#-----------------------------redirection for image upload for classification---------------------------------------
def classification_image_upload(request):
    if request.method == 'POST':
        form = classificationform(request.POST, request.FILES)
        uploaded_file=request.FILES
        if form.is_valid():
            model_instance = form.save(commit=False)
            model_instance.caption=uploaded_file['image'].name
            model_instance.uploaded_by=request.user.id
            form.save()
            return redirect('/classification')
    else:
        form = classificationform()
        return render(request, 'classification_upload.html', {'form' : form})
    print("",request.user.id)
    return render(request, 'classification_upload.html', {'form' : form})
            
#-----------------------------redirection for classification result page ---------------------------------------
def classification(request):
    pathstring=str
    up=Classification_Report.objects.last()
    print(up)
    img=Classification_Report.objects.filter(caption=up).last()
    index = ['Glioma','Meningioma','Normal','Adenoma']
    print("img : ",img)
    pathstring="media/classification/"+str(img)
    print(pathstring)
    model2=tf.keras.models.load_model("PROJECT/static/model.h5")
    test_image1 = load_img(pathstring,target_size = (224,224))
    print(test_image1)
    test_image1 = img_to_array(test_image1)
    test_image1 = np.expand_dims(test_image1,axis=0)
    result1 = np.argmax(model2.predict(test_image1/255.0),axis=1)
    print(index[result1[0]])
    img.tumor_type=index[result1[0]]
    img.save()
    return render(request, 'home/classification_display.html', {'Classification' : index[result1[0]],"img":img, 'media_url':settings.MEDIA_URL})

#-----------------------------redirection for Success response page---------------------------------------
def success(request):
    return HttpResponse('successfully uploaded')

#------------------------------fusion algorithm -------------------------------------------
class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()
    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 3:
                feature_maps.append(x)
        return feature_maps

class Fusion:

    def __init__(self, input):
        """
        Class Fusion constructor
        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.input_images = input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGG19(self.device)

    def fuse(self):
        """
        A top level method which fuse self.images
        """
        # Convert all images to YCbCr format
        self.normalized_images = [-1 for img in self.input_images]
        self.YCbCr_images = [-1 for img in self.input_images]
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx] = self._RGB_to_YCbCr(img)
                self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
            else:
                self.normalized_images[idx] = img / 255.
        # Transfer all images to PyTorch tensors
        self._tranfer_to_tensor()
        # Perform fuse strategy
        fused_img = self._fuse()[:, :, 0]
        # Reconstruct fused image given rgb input images
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx][:, :, 0] = fused_img
                fused_img = self._YCbCr_to_RGB(self.YCbCr_images[idx])
                fused_img = np.clip(fused_img, 0, 1)

        return (fused_img * 255).astype(np.uint8)
        # return fused_img

    def _fuse(self):
        """
        Perform fusion algorithm
        """
        with torch.no_grad():
            imgs_sum_maps = [-1 for tensor_img in self.images_to_tensors]
            for idx, tensor_img in enumerate(self.images_to_tensors):
                imgs_sum_maps[idx] = []
                feature_maps = self.model(tensor_img)
                for feature_map in feature_maps:
                    sum_map = torch.sum(feature_map, dim=1, keepdim=True)
                    imgs_sum_maps[idx].append(sum_map)
            max_fusion = None
            for sum_maps in zip(*imgs_sum_maps):
                features = torch.cat(sum_maps, dim=1)
                weights = self._softmax(F.interpolate(features,
                                        size=self.images_to_tensors[0].shape[2:]))
                weights = F.interpolate(weights,
                                        size=self.images_to_tensors[0].shape[2:])
                current_fusion = torch.zeros(self.images_to_tensors[0].shape)
                for idx, tensor_img in enumerate(self.images_to_tensors):
                    current_fusion += tensor_img * weights[:,idx]
                if max_fusion is None:
                    max_fusion = current_fusion
                else:
                    max_fusion = torch.max(max_fusion, current_fusion)

            output = np.squeeze(max_fusion.cpu().numpy())
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return output

    def _RGB_to_YCbCr(self, img_RGB):
            """
            A private method which converts an RGB image to YCrCb format
            """
            img_RGB = img_RGB.astype(np.float32) / 255.
            return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    def _YCbCr_to_RGB(self, img_YCbCr):
            """
            A private method which converts a YCrCb image to RGB format
            """
            img_YCbCr = img_YCbCr.astype(np.float32)
            return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def _is_gray(self, img):
            """
            A private method which returns True if image is gray, otherwise False
            """
            if len(img.shape) < 3:
                return True
            if img.shape[2] == 1:
                return True
            b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
            if (b == g).all() and (b == r).all():
                return True
            return False

    def _softmax(self, tensor):
            """
            A private method which compute softmax ouput of a given tensor
            """
            tensor = torch.exp(tensor)
            tensor = tensor / tensor.sum(dim=1, keepdim=True)
            return tensor

    def _tranfer_to_tensor(self):
            """
            A private method to transfer all input images to PyTorch tensors
            """
            self.images_to_tensors = []
            for image in self.normalized_images:
                np_input = image.astype(np.float32)
                if np_input.ndim == 2:
                    np_input = np.repeat(np_input[None, None], 3, axis=1)
                else:
                    np_input = np.transpose(np_input, (2, 0, 1))[None]
                if self.device == "cuda":
                    self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
                else:
                    self.images_to_tensors.append(torch.from_numpy(np_input)) 

#===========================================fusion_ct_mri==========================================

def fusion_ct_mri(request):

    
    # Load MRI image
    writ=WritingImage.objects.order_by('-writing').values()[:2]
    print(writ)
    a=list()
    b=dict()
    c=dict()
    for e in writ:
        a.append(e)
    print("dict ",a)
    b.update(a[0])
    c.update(a[1])
    img1=[]
    firstimage=list(b.values())
    secondimage=list(c.values())
    print(firstimage,secondimage)
    firstimage=firstimage[1]
    secondimage=secondimage[1]
    firstimagepath="media/"+firstimage
    secondimagepath="media/"+secondimage
    img1.append(firstimagepath)
    img1.append(secondimagepath)
    print("printing paths : ",img1[0])
    print("printing paths : ",img1[1])
    mri_image = cv2.imread(img1[0])
    mri_image = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)
    titles = ['Approximation', ' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(mri_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        path='PROJECT/media/fusion/mri_'+str(i)+'.jpg'
        cv2.imwrite(path,a)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    print("1")
    # Load CT Image
    ct_image = cv2.imread(secondimagepath)
    ct_image = cv2.cvtColor(ct_image, cv2.COLOR_BGR2GRAY)
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(ct_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        path='PROJECT/media/fusion/ct_'+str(i)+'.jpg'
        cv2.imwrite(path,a)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    print("2")
    img = Image.open(r"PROJECT/media/fusion/ct_0.jpg") 
    img = img.resize((300,300))
    img.save("PROJECT/media/fusion/ct_0.jpg")
    img = Image.open(r"PROJECT/media/fusion/mri_0.jpg") 
    img = img.resize((300,300))
    img.save("PROJECT/media/fusion/mri_0.jpg")
    print("3-resize")
    # Calling the methods for Siamese on LL Images
    input_images = []
    mri = cv2.imread('PROJECT/media/fusion/ct_0.jpg')
    mri = cv2.cvtColor(mri, cv2.COLOR_BGR2GRAY)
    ct = cv2.imread('PROJECT/media/fusion/mri_0.jpg')
    ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
    input_images.append(mri)
    input_images.append(ct)
    # Compute fusion image
    FU = Fusion(input_images)
    fusion_img = FU.fuse()
    # Write fusion image
    cv2.imwrite('PROJECT/static/finalimage.jpg', fusion_img)
    writing=WritingImage.objects.order_by('-writing').all()[:2]
    detection_tumor()
    return render(request, 'home/fusion_display.html',{'writing':writing,'media_url':settings.MEDIA_URL})
""" Application Of Image Fusion For Detecting Tumors
Username: 	
Password: 	
def fusion_upload(request):
    if request.method == 'POST':
        form = Fusionform(request.POST, request.FILES)
        uploaded_file=request.FILES.getlist('images','images2')
        print("dead : ",uploaded_file)
        if form.is_valid():
            model_instance = form.save(commit=False)
            model_instance.caption=uploaded_file['image'].name
            model_instance.caption2=uploaded_file['image2'].name
            form.save()
            print("123")
            return redirect('/fusion_ct_mri')
    else:
        form = Fusionform()
        return render(request, 'fusion_upload.html', {'form' : form})
    return render(request, 'fusion_upload.html', {'form' : form})
"""
def writing_create(request):

  """
  글작성
  """  
  if request.method == 'POST':
    form = WritingFullForm(request.POST, request.FILES)
    files = request.FILES.getlist('images')
    if form.is_valid():
      writing = form.save(commit=False)
      writing.create_date = timezone.now()
      writing.save()
      if files and writing:
        for f in files:
          WritingImage.objects.create(image=f)
        return redirect('/fusion_ct_mri')
  else:
    form = WritingFullForm()
  context = {'form': form}
  return render(request, 'fusion_upload.html', context)

def filters_noise(request):

    query= request.GET.get('q')
    query2= request.GET.get('p')
    query3= request.GET.get('r')
    button=request.GET.get('done')
    fetchingradiologistname=User.objects.filter(is_employee=1).values()
    radio=fetchingradiologistname
    my_list = []
    for i in range(0,int(len(radio))):
       radio=fetchingradiologistname[i]
       for j  in radio.values() :
          my_list.append(j)
    myusername=list()
    myid=list()
    if len(my_list)>=4:
       myusername.append(my_list[4])
    if len(my_list)>=18:
       myusername.append(my_list[18])
    if len(my_list)>=19:
      for i in range(19,len(my_list)):
          if (i)%(14)==0:
              myusername.append(my_list[i+4])
      myid.append(my_list[0])
      for i in range(18,len(my_list)):
          if (i)%(14)==0:
              myid.append(my_list[i])
    print(int(len(myusername)))
    mylist = zip(myusername, myid)
    #------------------------------------------------------------------------------
    patientname=User.objects.filter(is_customer=1).values()
    patient=patientname
    my_list2 = []
    for i in range(0,int(len(patient))):
       patient=fetchingradiologistname[i]
       for j  in patient.values() :
          my_list2.append(j)
    myusername2=list()
    myid2=list()
    if len(my_list2)>=4:
       myusername2.append(my_list2[4])
    if len(my_list2)>=18:
       myusername2.append(my_list2[18])
    if len(my_list2)>=19:
      for i in range(19,len(my_list2)):
          if (i)%(14)==0:
              myusername2.append(my_list[i+4])
    myid2.append(my_list2[0])
    for i in range(18,len(my_list2)):
        if (i)%(14)==0:
            myid2.append(my_list2[i])
    print(int(len(myusername2)))
    mylist2 = zip(myusername2, myid2)
    print("query : {} {} {}".format(query,query2,query3))
    if query!=None :
        lookups=   Q(caption__contains=query ) or Q(uploaded_by=query2 ) or Q(uploaded_by=query3 )
        results= Denoise.objects.filter(lookups).values()
        context={'results': results,
                     'submitbutton': button}
        print(results)
        return render(request, 'reports/reports.html',{'results':results,'radio':myusername,'radio2':myid,'radio3':myusername2,'radio4':myid2,'mylist':mylist,'mylist2':mylist2,'myid3':myusername})
    return render(request, 'reports/reports.html',{'radio':myusername,'radio2':myid,'radio3':myusername2,'radio4':myid2,'mylist':mylist,'mylist2':mylist2,'myid3':myusername})

def colouring(request):
    context={}
    image = cv2.imread('PROJECT/static/finalimage.jpg')
    query= request.GET.get('color')
    if query is None :
        return render(request,"colouring.html")
    elif query :
        print(query)
        lst_int = [int(x) for x  in query.split(",")]
        print(lst_int)
        # Change color to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        # Reshape image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image.reshape((-1,3))
        # Convert to float type
        pixel_vals = np.float32(pixel_vals)
        # define stopping criteria
        # you can change the number of max iterations for faster convergence!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        ## Select a value for k
        # then perform k-means clustering
        k = 3
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        labels_reshape = labels.reshape(image.shape[0], image.shape[1])
        plt.imshow(segmented_image)
        ##  Visualize one segment, try to find which is the leaves, background, etc!
        plt.imshow(labels_reshape==0, cmap='gray')
        # mask an image segment by cluster
        cluster = 0 # the first cluster
        masked_image = np.copy(image)
        # turn the mask green!
        masked_image[labels_reshape == cluster] = [lst_int[2],lst_int[1],lst_int[0]]
        cv2.imwrite("PROJECT/static/cluster1.jpg", masked_image)
        # mask an image segment by cluster
        cluster = 2 # the third cluster
        masked_image = np.copy(image)
        # turn the mask green!
        masked_image[labels_reshape == cluster] = [lst_int[2],lst_int[1],lst_int[0]]
        cv2.imwrite("PROJECT/static/cluster2.jpg", masked_image)
        print("1")
        context={'a':"a"}
    return render(request,"colouring.html",context)

def detection_tumor():
    img = cv2.imread('PROJECT/static/finalimage.jpg') #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    v += 20000
    final_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("image_processed.jpg", img)
    img_ini =  cv2.imread('PROJECT/static/finalimage.jpg')  # read image
    gray_img = cv2.cvtColor(img_ini,cv2.COLOR_BGR2GRAY) #convert image to gray
    plt.imshow(gray_img,cmap='gray')
    plt.title("initial image")
    plt.show()
    gray = cv2.medianBlur(gray_img,3) 
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU) #applying thresholding  to get the skull portion
    #cv2.imwrite("Threshold_img.jpg",thresh)
    plt.imshow(thresh ,cmap='gray')
    plt.savefig('PROJECT/static/detect3.png')
    plt.show()
    plt.savefig("output.jpg")
    import numpy as np
    colormask = np.zeros(img_ini.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255)) #overlaying mask over original image
    blended = cv2.addWeighted(img_ini,0.7,colormask,0.1,0)
    b,g,r = cv2.split(blended)
    rgb_img = cv2.merge([r,g,b])
    plt.imshow(rgb_img)
    plt.show()
    #cv2.imwrite("blended_img.jpg",rgb_img)
    ret, markers = cv2.connectedComponents(thresh) #finding the connected components in the image
    plt.imshow(markers)
    plt.savefig('PROJECT/static/detect4.png')
    plt.show()
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    largest_component = np.argmax(marker_area)+1  #finding the largest one which will be the brain                    
    brain_mask = markers==largest_component
    brain_out = img_ini.copy()
    brain_out[brain_mask==False] = (0,0,0) #filling rest of the background with black
    plt.imshow(brain_out)
    plt.show()
    img = brain_out
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # Make the feature vectors
    im = opening
    X = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)
    # Perform Clustering
    km = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    km.fit(X.astype(float)) # the .astype method is only to stop the .fit method
    # from throwing a warning.
    labels = np.reshape(km.labels_, im.shape[0:2])
    # Plotting results
    pl.figure()
    pl.imshow(img_ini)
    # Read image
    im = opening
    # Make the feature vectors
    # Make the feature vectors
    X = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)
    # Perform Clustering
    N_clus = 10
    km = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    km.fit(X.astype(float)) # the .astype method is only to stop the .fit method
    # from throwing a warning.
    labels = np.reshape(km.labels_, im.shape[0:2])
    # Plotting results
    pl.figure()
    pl.imshow(im)
    for l in range(N_clus):
       a= pl.contour(labels == 2, contours=1, \
                colors=[matplotlib.cm.nipy_spectral(l / float(N_clus)), ])
    cv2.imwrite("PROJECT/static/now.jpg",im)
    pl.savefig("PROJECT/static/detect1.jpg")
    im = Image.open('PROJECT/static/now.jpg')
    data = np.array(im)
    r1, g1, b1 = 157,157,157 # Original value
    r3, g3, b3 = 183,183,183 # Original value
    r2, g2, b2 = 45, 45, 45 # Value that we want to replace it with
    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red >r1) & (green > g1 ) & (blue > b1 )
    data[:,:,:3][mask] = [r2, g2, b2]
    im = Image.fromarray(data)
    im.save('PROJECT/static/fig1_modified.png')
    # Read image
    im = rgb_img
    # Make the feature vectors
    # Make the feature vectors
    X = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)
    # Perform Clustering
    N_clus = 10
    km = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    km.fit(X.astype(float)) # the .astype method is only to stop the .fit method
    # from throwing a warning.
    labels = np.reshape(km.labels_, im.shape[0:2])
    # Plotting results
    pl.figure()
    pl.imshow(im)
    for l in range(N_clus):
        a=pl.contour(labels == 2, contours=1, colors=[matplotlib.cm.nipy_spectral(l / float(N_clus)), ])
    plt.xticks(())
    plt.yticks(())
    plt.savefig("PROJECT/static/detect2.png")
    plt.show()
    pl.figure()
    pl.imshow(im)
    plt.savefig('image.jpg')
    images = [Image.open(x) for x in ['PROJECT/static/detect1.jpg','PROJECT/static/detect2.png']]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('PROJECT/static/test.jpg')
    images = [Image.open(x) for x in ['PROJECT/static/detect3.png','PROJECT/static/detect4.png']]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('PROJECT/static/test1.jpg')
    # read the images
    img1 = cv2.imread('PROJECT/static/test.jpg')
    img2 = cv2.imread('PROJECT/static/test1.jpg')
    # vertically concatenates images 
    # of same width 
    im_v = cv2.vconcat([img1,img2])
    # show the output image
    cv2.imwrite('PROJECT/static/final.jpg', im_v)

def classify_fusion(request):
    pathstring=str
    index = ['Glioma','Meningioma','Normal','Adenoma']
    model2=tf.keras.models.load_model("PROJECT/static/model.h5")
    test_image1 = load_img('PROJECT/static/finalimage.jpg',target_size = (224,224))
    print(test_image1)
    test_image1 = img_to_array(test_image1)
    test_image1 = np.expand_dims(test_image1,axis=0)
    result1 = np.argmax(model2.predict(test_image1/255.0),axis=1)
    print(index[result1[0]])
    tumor_type=index[result1[0]]
    print(tumor_type)
    return render(request, 'home/after_fusion.html', {'Classification' : index[result1[0]],'media_url':settings.MEDIA_URL})

