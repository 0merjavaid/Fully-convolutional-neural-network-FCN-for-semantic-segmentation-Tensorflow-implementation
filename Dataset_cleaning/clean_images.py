from PIL import Image
import cv2
import numpy as np
import os
import glob
from skimage import data, exposure
def clean():
    if not os.path.exists("Dataset"):

        os.makedirs("Dataset")
    os.chdir("../../IMAGES/Labels")
    """for file in glob.glob("*.*"):
        image=Image.open(file)
        name="../../Retina-segmentation-with-FCN/"+"Dataset/"+file.split(".")[0]+".png"
        print name
        image.save(name)"""

    for file in glob.glob("*.*"):
        image=Image.open(file)
        name="../../Retina-segmentation-with-FCN/"+"Labels/"+file.split(".")[0]+".png"
        print name
        image.save(name)

def enhance(im):
    copy=im.copy()
    B=im[:,:,0]
    G=im[:,:,1]
    R=im[:,:,2]
    eqB = exposure.equalize_adapthist(B, clip_limit=0.0051)*255
    eqG = exposure.equalize_adapthist(G, clip_limit=0.01)*255

    eqR = exposure.equalize_adapthist(R, clip_limit=0.0051)*255
    copy[:,:,0]=eqB
    copy[:,:,1]=eqG
    copy[:,:,2]=eqR
    return copy.astype("uint8")
def contrast(path):
    os.chdir(path)
    for file in glob.glob("*.png"):
        im=cv2.imread(file)
        enhanced=enhance(im)
        name="Enhanced/"+file
        cv2.imwrite(name,enhanced)



def rotation(input_image,input_label,output_image,output_label):
    os.chdir(input_image)
    for file in glob.glob("*.png"):
        im=Image.open(file)
        label_name=input_label+file
        label= Image.open(label_name)

        for i in range(360):
            rotated_image=im.rotate(i,Image.BICUBIC)
            rotated_label=label.rotate(i,Image.BICUBIC)
            rot_im_name=output_image+str(i)+file
            rot_label_name=output_label+str(i)+file

            rotated_image.save(rot_im_name)
            rotated_label.save(rot_label_name)
        break

rotation("/home/omer/Desktop/omer/retina/Retina-segmentation-with-FCN/Dataset/",
"/home/omer/Desktop/omer/retina/Retina-segmentation-with-FCN/Labels/",
"/home/omer/Desktop/omer/retina/Retina-segmentation-with-FCN/Train/",
"/home/omer/Desktop/omer/retina/Retina-segmentation-with-FCN/Train_label/")
