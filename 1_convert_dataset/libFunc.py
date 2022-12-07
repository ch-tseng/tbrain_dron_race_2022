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