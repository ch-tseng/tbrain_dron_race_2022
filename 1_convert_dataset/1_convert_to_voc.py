import os
import glob
import cv2
import numpy as np
from libFunc import *

dataset_path = r'train'
class_map = { '0':'car', '1':'hov', '2':'person', '3':'motorcycle' }
output_voc_path = r'voc_ds'

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#------------------------------------------------

dataset_path = dataset_path.replace('\\', '/')
output_voc_path = output_voc_path.replace('\\', '/')

label_files = glob.glob( os.path.join(dataset_path, '*.txt' ) )
img_files = glob.glob( os.path.join(dataset_path, '*.png' ) )

if not os.path.exists(output_voc_path):
    os.makedirs(output_voc_path)
    
if not os.path.exists( os.path.join(output_voc_path, 'labels') ):
    os.makedirs(os.path.join(output_voc_path, 'labels'))
    
if not os.path.exists( os.path.join(output_voc_path, 'images') ):
    os.makedirs(os.path.join(output_voc_path, 'images'))    

obj_count = {}
#-------------------------------------------------

for cfile in label_files:
    file_bname = os.path.basename(cfile)
    with open(cfile, 'r') as f:
        lines = f.readlines()
    
    bbox_objects = {}
    for line in lines:
        line = line.replace('\n', '')
        datas = line.split(',')
        if len(datas)==5:
            class_id = datas[0].strip()
            if class_id in class_map:
                class_name = class_map[class_id]
            else:
                print('Cannot find the class:{} --->{}'.format(class_id, line))
                
            x, y, w, h = int(datas[1]), int(datas[2]), int(datas[3]), int(datas[4])
            
            if(class_name in obj_count):
                counts = obj_count[class_name]+1
            else:
                counts = 1
                
            obj_count.update({class_name:counts})

            if(class_name in bbox_objects):
                bbox_objects[class_name].append([x,y,w,h])
            else:
                bbox_objects[class_name] = [[x,y,w,h]]
                
        else:
            print("error:{}, skip it.".format(line))
                
    if(len(bbox_objects)>0):
        png_path = cfile.replace('.txt', '.png')
        img = cv2.imread(png_path)       
        
        makeLabelFile(img, bbox_objects, file_bname, xml_file, object_xml_file, output_voc_path)
            
        


