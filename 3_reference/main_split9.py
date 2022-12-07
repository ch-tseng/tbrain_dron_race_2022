from libDNNYolo import opencvYOLO
import cv2
import imutils
import time, os, glob
import random
import numpy as np
from configparser import ConfigParser
import ast
from libFunc import *

tcolors = [ (0,0,255), (0,255,0), (255,0,255), (255,255,0) ]
class_id_map = { 'car':0, 'hov':1, 'person':2, 'motorcycle':3 }

cfg = ConfigParser()
cfg.read("demo_split9.ini",encoding="utf-8")

#medias path for detection
media = cfg.get("Media", "media")
title = cfg.get("Media", "title")
display_width = cfg.getint("Media", "display_width")
output_predict = cfg.getboolean("Media", "output_predict")
video_out= cfg.get("Media", "video_out")
framerate = cfg.getint("Media", "framerate")
output_folder = cfg.get("Media", "output_folder")
resize_video = ast.literal_eval(cfg.get("Media", "resize_video"))

media = media.replace('\\', '/')

crop_org = cfg.getboolean("Detect", "crop_org")
crop_pad = cfg.getint("Detect", "crop_pad")
make_dataset = cfg.getboolean("Detect", "make_dataset")
make_dataset_path = cfg.get("Detect", "make_dataset_path")
if make_dataset is True:
    xml_file = "xml_file.txt"
    object_xml_file = "xml_object.txt"
    
    if not os.path.exists('make_dataset_path'):
        os.makedirs('make_dataset_path')
        
    if not os.path.exists( os.path.join(make_dataset_path, 'labels') ):
        os.makedirs(os.path.join(make_dataset_path, 'labels'))
    
    if not os.path.exists( os.path.join(make_dataset_path, 'images') ):
        os.makedirs(os.path.join(make_dataset_path, 'images')) 

cvs_output_file = cfg.get("Detect", "cvs_output_file")
person_split = cfg.getint("Model_person", "split")
vehicle_split = cfg.getint("Model_vehicle", "split")
#----------------------------------------------------------------------
#human model configuration
path_weights = cfg.get("Model_person", "path_weights")
path_objname = cfg.get("Model_person", "path_objname")
path_darknetcfg = cfg.get("Model_person", "path_darknetcfg")

path_objname = path_objname.replace('\\', '/')
path_weights = path_weights.replace('\\', '/')
path_darknetcfg = path_darknetcfg.replace('\\', '/')

if path_weights[-2:] == 'pt':
    mtype = 'yolov5'
    
else:
    mtype = 'darknet'

model_person = opencvYOLO( \
        mtype=mtype, imgsize=ast.literal_eval(cfg.get("Model_person", "model_size")), \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg=path_darknetcfg, \
        score=float(cfg.get("Model_person", "confidence")), \
        nms=float(cfg.get("Model_person", "nms")), \
        tcolors=ast.literal_eval(cfg.get("Model_person", "tcolors")), \
        gpu=cfg.getboolean("Model_person", "gpu"))
        
person_classes = ast.literal_eval(cfg.get("Model_person", "classes"))        
        
#----------------------------------------------------------------------        
#vehicle model configuration
path_weights = cfg.get("Model_vehicle", "path_weights")
path_objname = cfg.get("Model_vehicle", "path_objname")
path_darknetcfg = cfg.get("Model_vehicle", "path_darknetcfg")

path_objname = path_objname.replace('\\', '/')
path_weights = path_weights.replace('\\', '/')
path_darknetcfg = path_darknetcfg.replace('\\', '/')

if path_weights[-2:] == 'pt':
    mtype = 'yolov5'
    
else:
    mtype = 'darknet'

model_vehicle = opencvYOLO( \
        mtype=mtype, imgsize=ast.literal_eval(cfg.get("Model_vehicle", "model_size")), \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg=path_darknetcfg, \
        score=float(cfg.get("Model_vehicle", "confidence")), \
        nms=float(cfg.get("Model_vehicle", "nms")), \
        tcolors=ast.literal_eval(cfg.get("Model_vehicle", "tcolors")), \
        gpu=cfg.getboolean("Model_vehicle", "gpu"))        

vehicle_classes = ast.literal_eval(cfg.get("Model_vehicle", "classes"))  
#---------------------------------------------------------------------- 
if os.path.isdir(media):
    img_list = glob.glob(os.path.join(media, '*.png'))
    sorted(img_list)

    media_type = 'imgs'
    INPUT = img_list
else:
    media_type = 'video'
    INPUT = cv2.VideoCapture(media)
    framerate = INPUT.get(cv2.CAP_PROP_FPS)

start = time.time()
last_time = time.time()
last_frames = 0

last_time, last_frames, fps = time.time(), 0, 0

def get_img():
    global frameID

    frame = None
    hasFrame = False
    frame_name = None
    if media_type == 'video':
        hasFrame, frame = INPUT.read()
        frame_name = None
        if frame is not None and resize_video is not None:
            frame = cv2.resize(frame, resize_video)

    else:
        if frameID>=len(INPUT):
            hasFrame = False
            frame_name = None

        else:
            hasFrame = True
            frame = cv2.imread(INPUT[frameID])
            frame_name = os.path.basename(INPUT[frameID])
            

    return hasFrame, frame, frame_name

def fps_count(total_frames):
    global last_time, last_frames, fps

    timenow = time.time()

    if(timenow - last_time)>6:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        last_time  = timenow
        last_frames = total_frames

    return fps


if __name__ == "__main__":
    print('Push Q to quit the program.')
    if not os.path.exists('output'):
        os.makedirs('output')

    #record video   
    if(video_out!=""):
        #video_out = os.path.join('output', video_out)

        if media_type == 'video':
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

        else:
            img_video_size = (1920, 1080)
            width = img_video_size[0]
            height = img_video_size[1]
            framerate = 0.5

        if(output_predict is True):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            if resize_video is not None:
                out = cv2.VideoWriter( os.path.join(output_folder,video_out), fourcc, framerate, resize_video)
                #print(os.path.join(output_folder,video_out), resize_video)
            else:
                out = cv2.VideoWriter(os.path.join(output_folder,video_out), fourcc, framerate, (int(width),int(height)))

    frameID = 0
    hasFrame, frame, frame_name = get_img()
    print('media_type', media_type)   
    
    
    while hasFrame:
        #print('--------->', frame_name)
        img = frame.copy()
        counts, bbox_objects, obj_count = 0, {}, {}
        (oh, ow, _) = img.shape
        final_bboxes, final_labels, final_scores, final_cids, final_scores = [], [], [], [], []
        
        if crop_org is True:

            if person_split == 4:
                img1 = img[0:int(oh/2)+crop_pad, 0:int(ow/2)+crop_pad]
                img2 = img[0:int(oh/2)+crop_pad, int(ow/2)-crop_pad:ow]
                img3 = img[int(oh/2)-crop_pad:oh, 0:int(ow/2)+crop_pad]
                img4 = img[int(oh/2)-crop_pad:oh, int(ow/2)-crop_pad:ow]

                #print('reference img1 for person')
                img1_person = model_person.getObject(img1, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append(box)
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img2 for person')
                img2_person = model_person.getObject(img2, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/2)-crop_pad, int(oh/2)-crop_pad
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1], box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img3 for person')
                img3_person = model_person.getObject(img3, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/2)-crop_pad, int(oh/2)-crop_pad
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0], box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img4 for person')
                img4_person = model_person.getObject(img4, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/2)-crop_pad, int(oh/2)-crop_pad
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )

            elif person_split == 9:
                img1 = img[0:int(oh/3)+crop_pad, 0:int(ow/3)+crop_pad]
                img2 = img[0:int(oh/3)+crop_pad, int(ow/3):2*int(ow/3)+crop_pad]
                img3 = img[0:int(oh/3)+crop_pad, 2*int(ow/3):ow]
                img4 = img[int(oh/3):2*int(oh/3)+crop_pad, 0:int(ow/3)+crop_pad]
                img5 = img[int(oh/3):2*int(oh/3)+crop_pad, int(ow/3):2*int(ow/3)+crop_pad]
                img6 = img[int(oh/3):2*int(oh/3)+crop_pad, 2*int(ow/3):ow]
                img7 = img[2*int(oh/3):oh, 0:int(ow/3)+crop_pad]
                img8 = img[2*int(oh/3):oh, int(ow/3):2*int(ow/3)+crop_pad]
                img9 = img[2*int(oh/3):oh, 2*int(ow/3):ow]

                #print('reference img1 for person')
                img1_person = model_person.getObject(img1, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append(box)
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img2 for person')
                img2_person = model_person.getObject(img2, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3), int(oh/3)
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1], box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )   
                #print('reference img3 for person')
                img3_person = model_person.getObject(img3, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3)*2, int(oh/3)
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1], box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] ) 
                    final_scores.append( model_person.scores[i] )
                #print('reference img4 for person')
                img4_person = model_person.getObject(img4, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = 0, int(oh/3)
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0], box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img5 for person')
                img5_person = model_person.getObject(img5, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3), int(oh/3)
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img6 for person')
                img6_person = model_person.getObject(img6, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3)*2, int(oh/3)
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img7 for person')
                img7_person = model_person.getObject(img7, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = 0, int(oh/3)*2
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0], box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img8 for person')
                img8_person = model_person.getObject(img8, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3), int(oh/3)*2
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )
                #print('reference img9 for person')
                img8_person = model_person.getObject(img9, drawBox=False, char_type='Chinese')
                for i, box in enumerate(model_person.bbox):
                    add_x, add_y = int(ow/3)*2, int(oh/3)*2
                    final_cids.append( class_id_map[model_person.labelNames[i]] )
                    final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                    final_labels.append( model_person.labelNames[i] )
                    final_scores.append( model_person.scores[i] )

                #--------------------------------------------------------------------------

            if vehicle_split == 4:
                img1 = img[0:int(oh/2)+crop_pad, 0:int(ow/2)+crop_pad]
                img2 = img[0:int(oh/2)+crop_pad, int(ow/2)-crop_pad:ow]
                img3 = img[int(oh/2)-crop_pad:oh, 0:int(ow/2)+crop_pad]
                img4 = img[int(oh/2)-crop_pad:oh, int(ow/2)-crop_pad:ow]

            else:
                img1 = img[0:int(oh/3)+crop_pad, 0:int(ow/3)+crop_pad]
                img2 = img[0:int(oh/3)+crop_pad, int(ow/3):2*int(ow/3)+crop_pad]
                img3 = img[0:int(oh/3)+crop_pad, 2*int(ow/3):ow]
                img4 = img[int(oh/3):2*int(oh/3)+crop_pad, 0:int(ow/3)+crop_pad]
                img5 = img[int(oh/3):2*int(oh/3)+crop_pad, int(ow/3):2*int(ow/3)+crop_pad]
                img6 = img[int(oh/3):2*int(oh/3)+crop_pad, 2*int(ow/3):ow]
                img7 = img[2*int(oh/3):oh, 0:int(ow/3)+crop_pad]
                img8 = img[2*int(oh/3):oh, int(ow/3):2*int(ow/3)+crop_pad]
                img9 = img[2*int(oh/3):oh, 2*int(ow/3):ow]


            #print('reference img1 for vehicle')
            img1_vehicle = model_vehicle.getObject(img1, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append(box)
                final_labels.append( model_vehicle.labelNames[i] )
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img2 for vehicle')
            img2_vehicle = model_vehicle.getObject(img2, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):                
                add_x, add_y = int(ow/3), 0
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1], box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])    
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img3 for vehicle')
            img3_vehicle = model_vehicle.getObject(img3, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = int(ow/3)*2, 0
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1], box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])   
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img4 for vehicle')
            img4_vehicle = model_vehicle.getObject(img4, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = 0, int(oh/3)
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0], box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img5 for vehicle')
            img5_vehicle = model_vehicle.getObject(img5, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = int(ow/3), int(oh/3)
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img6 for vehicle')
            img6_vehicle = model_vehicle.getObject(img6, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = int(ow/3)*2, int(oh/3)
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img7 for vehicle')
            img7_vehicle = model_vehicle.getObject(img7, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = 0, int(oh/3)*2
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0], box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img8 for vehicle')
            img8_vehicle = model_vehicle.getObject(img8, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = int(ow/3), int(oh/3)*2
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )
            #print('reference img9 for vehicle')
            img9_vehicle = model_vehicle.getObject(img9, drawBox=False, char_type='Chinese')
            for i, box in enumerate(model_vehicle.bbox):
                add_x, add_y = int(ow/3)*2, int(oh/3)*2
                final_cids.append( class_id_map[model_vehicle.labelNames[i]] )
                final_bboxes.append( (box[0]+add_x, box[1]+add_y, box[2], box[3]) )
                final_labels.append(model_vehicle.labelNames[i])
                final_scores.append( model_vehicle.scores[i] )

            nms_indices = cv2.dnn.NMSBoxes(final_bboxes, final_scores, 0.25, 0.25)
            nms_scores, nms_labels, nms_boxes, nms_cids = [], [], [], []
            for ii in nms_indices:
                try:
                    i = ii[0]
                except:
                    i = ii

                box = final_bboxes[i]
                left = final_bboxes[i][0]
                top = final_bboxes[i][1]
                width = final_bboxes[i][2]
                height = final_bboxes[i][3]

                nms_scores.append(final_scores[i])
                nms_labels.append(final_labels[i])
                nms_boxes.append(final_bboxes[i])
                nms_cids.append(final_cids[i])


                txt_color = tcolors[final_cids[i]]
                cv2.rectangle(img, (left, top), (left+width, top+height), txt_color, 2)
                
                if make_dataset is True:
                    if(final_labels[i] in obj_count):
                        counts = obj_count[final_labels[i]]+1
                    else:
                        counts = 1
                        
                    obj_count.update({final_labels[i]:counts})
                    
                    if(final_labels[i] in bbox_objects):
                        bbox_objects[final_labels[i]].append([left,top,width,height])
                    else:
                        bbox_objects[final_labels[i]] = [[left,top,width,height]]

            nms_labels, nms_boxes = remove_error_box(nms_labels, nms_boxes, img)
            #cv2.imshow('result', imutils.resize(frame, height=800))
            #cv2.waitKey(1)
            cv2.imwrite( os.path.join(output_folder, frame_name), img)
            
            with open(cvs_output_file, 'a') as f:
                filename = frame_name.split('.')[0]
                for i, box in enumerate(nms_boxes):
                    f.write("{},{},{},{},{},{}\n".format(filename, class_id_map[nms_labels[i]], box[0], box[1], box[2], box[3] ))
            
            if make_dataset is True:            
                if(len(bbox_objects)>0):
                    makeLabelFile(frame, bbox_objects, frame_name, xml_file, object_xml_file, make_dataset_path)
            
        else:
            img = model.getObject(img, score, nms, drawBox=True, char_type='Chinese')

            if output_predict is True: out.write(frame)
        
        
            
        frameID += 1
        hasFrame, frame, frame_name = get_img()
        
    if output_predict is True: out.release()
        
    '''
        cv2.putText(img,  '2022/11/01', (36,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow('test', imutils.resize(img, width=display_width))
        #if media_type == 'imgs':
        #    cv2.imwrite( os.path.join(output_folder, str(frameID)+'_1.jpg'), img)


        if output_predict is True:
            if media_type == 'video':
                out.write( img )
                print('shape', img.shape)
                canvas = img
            else:
                canvas = np.zeros((1080, 1920, 3), dtype = 'uint8')
                img = imutils.resize( img, height=1080)
                if img.shape[1]>1920: img = imutils.resize( img, width=1920)
                ww, hh = img.shape[1]/2 , img.shape[0]/2
                x1, y1 = int(1920/2 - ww), int(1080/2 - hh)
                x2, y2 = x1 + img.shape[1], y1 + img.shape[0]
                canvas[y1:y2, x1:x2] = img
                out.write(canvas)

                if media_type == 'imgs':
                    cv2.imwrite( os.path.join(output_folder, str(frameID)+'_2.jpg'), canvas)


        cv2.imshow('test', imutils.resize(canvas, width=display_width))
        if media_type == 'video':
            k = cv2.waitKey(1)
        else:
            k = cv2.waitKey(1)

        if(k==113):
            break

        hasFrame, frame, frame_name = get_img()
        if hasFrame is False: break
        frameID += 1

        fps = fps_count(frameID)
        print('FPS', fps)

    if output_predict is True: out.release()
    '''
