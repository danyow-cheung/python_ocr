'''
https://www.youtube.com/watch?v=NApYP_5wlKY
Automatic Number Plate Recognition with opencv and easyocr
算法步骤
	使用边缘检测检测边框之后，添加蒙版再使用easyocr提取数据
'''

import easyocr 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import imutils 

path = 'image1.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)

# apply filter and find edges for localization
bfilter = cv2.bilateralFilter(gray,11,17,17)
edged = cv2.Canny(bfilter,30,200) # edge detection
# plt.imshow(edged)
# plt.show()

keypoints = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]
# print(contours)
location =None 
for contour in contours:
	approx = cv2.approxPolyDP(contour,10,True)
	if len(approx)==4:
		location = approx
		break 
print(location)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[location],0,255,-1)
new_image = cv2.bitwise_and(img,img,mask=mask)
# plt.imshow(new_image)
# plt.show()

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# plt.imshow(cropped_image)
# plt.show()
'''
模型在线的？
CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.
Downloading detection model, please wait. This may take several minutes depending upon your network connection.
'''
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()
