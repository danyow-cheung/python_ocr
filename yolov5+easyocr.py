'''
使用别人训练好的模型，直接进行推理
2023.4.11 danyow done
'''
import torch 
import matplotlib.pyplot as plt 
import numpy as np
import os 
import cv2 
import easyocr 

# 遇到FileNotFoundError: [Errno 2] No such file or directory: 'yolov5/hubconf.py'
# load函数的第一个参数修改为绝对路径
model = torch.hub.load('/Users/danyow/Desktop/ANPR/yolov5', 'custom', path='best.pt', source="local",force_reload=True)
# 定义识别图片路径
data_paths = ['/Users/danyow/Desktop/ANPR/carLicensePlate2.jpeg','/Users/danyow/Desktop/ANPR/carLicensePlate1.jpeg']
font = cv2.FONT_HERSHEY_SIMPLEX


for i in range(len(data_paths)):
    # 首先识别到是车牌
    results = model(data_paths[i])
    # 得到边框数值，用ocr
    results_df = results.pandas().xyxy[0].loc[0]

    x_min = int(results_df['xmin'])
    x_max = int(results_df['xmax'])
    y_min = int(results_df['ymin'])
    y_max = int(results_df['ymax'])
    label = results_df['name']
    conf = results_df['confidence']
    
    print(label,conf)
    if label =='License Plate':
        print('使用ocr进行识别')
        img = cv2.imread(data_paths[i])
        number_plate = img[y_min:y_max,x_min:x_max]  
        reader = easyocr.Reader(['en'],gpu=False)
        reader_res = reader.readtext(number_plate)
        top_left = tuple(reader_res[0][0][0])
        bottom_right = tuple(reader_res[0][0][2])

        text = reader_res[0][1]
        score = reader_res[0][2]
        print('OCR识别',text,score)
        
        # img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),5)
        # img = cv2.putText(img,text,top_left,font,1,(255,0,255),2,cv2.LINE_AA)
        
        # cv2.show(img)

