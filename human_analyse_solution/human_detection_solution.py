# Modules

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from threading import Thread
from queue import Queue

import numpy as np
import imutils
import cv2
import os
import time
import requests
import json



# Initialization MobileSSD class 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# MobileSSD Caffe Model initialization
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Directory for watchdog to observe
motion_path = "/var/www/vhosts/uni.aleksandrov.app/httpdocs/motion_detector/uploads/"

# Load threhsold settings from json config
with open('config.json', 'r') as f:
    config = json.load(f)

#store confidence threshold
confidence_threshold = config['detection']['confidence_rate']

# Queue for watchdog observer
path_queue = Queue()


# Settings for watchdog
if __name__ == "__main__":
    patterns = "*"
    ignore_patterns = ""
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)





def analyse_human(frame): # analyses an image for human detection and returns result

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    max_confidence = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15: #limited to person classes
            if confidence > confidence_threshold: #confidence settings
                if confidence > max_confidence: # finding the highest confidence
                    max_confidence = float(confidence)
    

    return float(str(max_confidence)[:4]) #return confidence
                
                




def deleteImage(path): # removes an image based on path 
    os.remove(path)


def selectImage(file_name): 
    image_location = motion_path+file_name # file location

    frame = cv2.imread(image_location) #stores image to variable
    deleteImage(image_location) #removes image
    
    return frame #return the image


def image_observer():

    analysed_path = "/var/www/vhosts/uni.aleksandrov.app/httpdocs/human_detection/analyzed/" #directory to place detected images

    while True:
        image_path = path_queue.get() #get a path from the queue
        file_name = os.path.basename(image_path) #find filename based on path

        frame = selectImage(file_name) # removes image from directory and stores in variable

        confidence = analyse_human(frame) # find confidence for image

        if confidence != 0: # if the confidence is not 0
            cv2.imwrite(analysed_path+file_name, frame) # store image in the analysed directory
            #send POST request to PHP api
            requests.post("https://uni.aleksandrov.app/api/cloud_server/image_analysed.php", data={'confidence': confidence, 'file_name' : file_name}) 


        print(confidence) #output confidence
        


# worker thread on image observer
worker = Thread(target=image_observer)
worker.setDaemon(True)
worker.start()



def on_created(event): # on new image in directory run this function
    path_queue.put(event.src_path) #add the full file path to the new image
    
    

    


my_event_handler.on_created = on_created #add event to handler



#start the observer to observe the directory where motions are uploaded
go_recursively = True
my_observer = Observer()
my_observer.schedule(my_event_handler, motion_path, recursive=go_recursively) 
my_observer.start() 


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    my_observer.stop()
    my_observer.join()