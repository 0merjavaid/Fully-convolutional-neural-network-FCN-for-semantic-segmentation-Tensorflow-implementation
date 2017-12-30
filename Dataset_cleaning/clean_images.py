from PIL import Image
import cv2
import numpy as np
import os
import glob
if not os.path.exists("Dataset"):
    os.makedirs("Dataset")
os.chdir("../../IMAGES/Dataset")
for file in glob.glob("*.*"):
    image=Image.open(file)
    name="../../Retina-segmentation-with-FCN/"+"Dataset/"+file.split(".")[0]+".png"
    print name
    image.save(name)
