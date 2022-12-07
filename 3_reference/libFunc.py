import os
import numpy as np
import cv2

def writeObjects(label, bbox, xml_file, object_xml_file):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(img, filename, fullpath, bboxes, xml_file, object_xml_file):
    xmlObject = ""

    for labelName, bbox_array in bboxes.items():
        for bbox in bbox_array:
            xmlObject = xmlObject + writeObjects(labelName, bbox, xml_file, object_xml_file)

    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(img, bboxes, img_file_name, xml_file, object_xml_file, output_dataset):
    #img_path = os.path.join(target_dataset, 'images')
    file_name, ext_name = os.path.splitext(img_file_name)
    jpgFilename = file_name + '.png'    
    xmlFilename = file_name + ".xml"

    output_xml_file = os.path.join(output_dataset, 'labels', xmlFilename)   
    xmlContent = generateXML(img, xmlFilename, output_xml_file, bboxes, xml_file, object_xml_file)
    file = open(output_xml_file, "w")
    file.write(xmlContent)
    file.close    

    cv2.imwrite(os.path.join(output_dataset, 'images', jpgFilename) , img)

def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def interArea(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / boxAArea

    return iou


def remove_error_box(labels, bboxes, scores, img):
    (height, width, _) = img.shape
    f_bboxes, f_labels, f_scores = [], [], []
    for i, box in enumerate(bboxes):
        dtype = labels[i]
        (bx1,by1,bw1,bh1) = box

        if dtype == 'person':
            if bh1<(height/3) and bw1<(width/4):
                f_bboxes.append(box)
                f_labels.append(dtype)
                f_scores.append(scores[i])

        else:
            remove = False

            if bh1>(height/2) or bw1>(width/3):
                remove = True

            else:
                for ii, b in enumerate(bboxes):
                    (bx2,by2,bw2,bh2) = b
                    if i!=ii:
                        iou = interArea( (bx1,by1,bx1+bw1,by1+bh1), (bx2,by2,bx2+bw2,by2+bh2) )
                        if dtype=='car' and labels[ii]=='car':
                            #if iou>0.9 or (bx1>=bx2 and by1>=by2 and (bx1+bw1)<=(bx2+bw2) and (by1+bh1)<=(by2+bh2)):
                            if (bx1>=bx2 and by1>=by2 and (bx1+bw1)<=(bx2+bw2) and (by1+bh1)<=(by2+bh2)):
                                remove = True
                                break

                if remove is False:
                    f_bboxes.append(box)
                    f_labels.append(dtype)
                    f_scores.append(scores[i])
                else:
                    print("remove", dtype, box)

    return f_labels, f_bboxes, f_scores
