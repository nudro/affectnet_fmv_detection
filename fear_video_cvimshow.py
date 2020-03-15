#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 13, 2020 coronavirus starts

@author: 559048
"""
from __future__ import print_function
import net
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import dlib
import imgaug as ia
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",  required=True, help="path to input video file")
args = vars(ap.parse_args())

#LOAD MODEL AND MOVE TO CUDA
net = net.Net()
net.load_state_dict(torch.load("fear5.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

#DETECTORS
haar_face = "haarcascade_frontalface_alt.xml"

faceCascade = cv2.CascadeClassifier(haar_face)

detector = dlib.get_frontal_face_detector() #detects faces

def img_processor(frame_rgb, x, y, w, h):
    im_size = 100
    horizontal_offset = 0 
    vertical_offset = 0 
    extracted_face = frame_rgb[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
    extracted_face_r = cv2.resize(extracted_face, (im_size, im_size)) #resize the extracted face
    extracted_face_arr = np.array(extracted_face_r)
    normalized = np.array(extracted_face_arr, np.float32) / 255.
    normalized_arr = np.expand_dims(normalized, axis=0) #only one pic, one sample, so expand dims to turn into 4D array
    face_torch = torch.from_numpy(normalized_arr) #pytorch will only accept torch tensors
    face_torch = face_torch.reshape(1, 3, im_size, im_size) #need to reshape from (1, 100,100, 3) to (1, 3, 100, 100)
    face_torch = face_torch.to(device)
    return face_torch

def prediction(test_image):
    with torch.no_grad():
        net.cuda()
        output = net(test_image)
        _, predicted = torch.max(output.data, 1)
        sm = torch.nn.Softmax()
        probabilities = sm(output) 
        fear_probs = probabilities[:, 0]
        neutral_probs = probabilities[:, 1]
        return fear_probs.item(), neutral_probs.item(), predicted.item()


#cv2.destroyAllWindows() #do not UNcomment this on the GPU cluster - X server not enabled
print("Starting up video stream...")
vc = cv2.VideoCapture(args["video"])

length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
size = (
    int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('video_predictions.avi', codec, 60.0, size)

frame_number = 0

if vc.isOpened():
    rval , frame = vc.read()

else:
    rval = False
    
with open('fear.csv', 'w', newline='') as csvfile:
    outfile = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    frame_counter = 1
    while rval:
    
        rval, frame = vc.read()

        # resize frame for speed.
        frame = cv2.resize(frame, (400,250), interpolation=cv2.INTER_CUBIC) 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    
        faces = faceCascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

        # face detection.
        nfaces = 0
        avg_fear = []
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            nfaces = nfaces + 1
        
            extracted_array = img_processor(frame_rgb, x, y, w, h) 
            extracted_array.to(device)
            p_fear, p_neutral, predicted_class = prediction(extracted_array.to(device)) #apply img_processor func: extracts face and resizes, converts to torch
            print(p_fear, p_neutral, predicted_class)
        
    
            # annotate main image with a label
            if predicted_class == 0:
                cv2.putText(frame, ("{:.2f}".format(p_neutral) + "Fear" ), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            elif predicted_class == 1:
                cv2.putText(frame, ("{:.2f}".format(p_fear) + "Neutral" ), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            print('Probability this is FEAR: {}'.format(p_fear))
             #add the probablity of fear to an array
            avg_fear.append(p_fear)
       
    
        #calculate the avg from that array
        avg_fear_value = np.round(np.mean(avg_fear), decimals = 4)
        #annotate average fear on frame
        cv2.putText(frame, ("{:.2f}".format(avg_fear_value) + "Crowd Fear" ), (280,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
        print(avg_fear_value)
        #save holder to file
        frame_counter += 1
        outfile.writerow((frame_counter, avg_fear_value)) 
    

        # show result
        cv2.imshow("Result",frame)
    
        #frame = cv2.resize(frame, (300,500))
        output.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # When everything is done, release the capture
    vc.release()
    cv2.destroyAllWindows()
