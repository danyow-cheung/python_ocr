# -*- coding: utf-8 -*-
"""Tesseract_OCR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/141i0UGbMpHWjSc7c8BT09IHV60Csldgl
"""

!pip install pytesseract

!sudo apt install tesseract-ocr

!sudo apt install libtesseract-dev

'''
https://www.youtube.com/watch?v=PY_N1XdFp4w
'''
import pytesseract 
import PIL.Image 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np

myconfig = r'--psm 6 --oem 3'

text = pytesseract.image_to_string(PIL.Image.open('stop.jpeg'),config=myconfig)

text

img = cv2.imread('stop.jpeg')
h,w,c = img.shape 
print(h,w,c)

from pytesseract import Output
data = pytesseract.image_to_data(img,config=myconfig,output_type = Output.DICT)
data

data['text']

amount_of_box = len(data['text'])
for i in range(amount_of_box):
  if float(data['conf'][i])>30:# 大于阈值
    (x,y,width,height) = (data['left'][i],data['top'][i],data['width'][i],data['height'][i])
    img = cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),2)
    img = cv2.putText(img,data['text'][i],(x,y+height+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),cv2.LINE_AA)
plt.imshow(img)
plt.show()

# boxes = pytesseract.image_to_boxes(img,config=myconfig)
# print(boxes)

for box in boxes.splitlines():
  box = box.split(" ")
  img = cv2.rectangle(img,(int(box[1]),h-int(box[2])),(int(box[3]),w-int(box[4])),(0,255,0))

plt.imshow(img)
plt.show()

