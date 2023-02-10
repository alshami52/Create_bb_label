import pandas as pd
import numpy as np
import cv2
import os


df = pd.read_csv("/scratch/datasets/UMD_triplet/OND.valtests.0000.0001_single_df.csv", index_col=None)
root = "/scratch/datasets/UMD_triplet/"
output_directory = "./boxes/"



for index, row in df.iterrows():
    address = row["new_image_path"]
    image_width = row["image_width"]
    image_height = row["image_height"]
    subject_ymin = row["subject_ymin"]
    subject_xmin = row["subject_xmin"]
    subject_ymax = row["subject_ymax"]
    subject_xmax = row["subject_xmax"]
    object_ymin = row["object_ymin"]
    object_xmin = row["object_xmin"]
    object_ymax = row["object_ymax"]
    object_xmax = row["object_xmax"]
    subject_name = row["subject_name"]
    subject_id = row["subject_id"]
    object_name = row["object_name"]
    object_id = row["object_id"]
    verb_name = row["verb_name"]
    verb_id = row["verb_id"]
    
    img = cv2.imread(root + address, 1)
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    #print(address, image_width, image_height, width, height, channels)
    #print(address, subject_xmin, subject_xmax, width, subject_ymin, subject_ymax, height)
    #print(address, object_xmin, object_xmax, width, object_ymin, object_ymax, height)
    assert height == image_height
    assert width == image_width
    assert subject_xmin <= subject_xmax
    assert subject_xmax <= width
    assert object_xmin <= object_xmax
    assert object_xmax <= width
    assert subject_ymin <= subject_ymax
    assert subject_ymax <= height
    assert object_ymin <= object_ymax
    assert object_ymax <= height
    
    if (subject_xmax >=0) or (subject_ymax >=0):
      subject_xmin = max(0,subject_xmin)
      subject_xmax = max(0,subject_xmax)
      subject_ymin = max(0,subject_ymin)
      subject_ymax = max(0,subject_ymax)
      im_1 = cv2.rectangle(img, (subject_xmin,subject_ymin), (subject_xmax,subject_ymax), (0,0,255), 3)
    else:
      im_1 = img

    if (object_xmax >=0) or (object_ymax >=0):
      object_xmin = max(0,object_xmin)
      object_xmax = max(0,object_xmax)
      object_ymin = max(0,object_ymin)
      object_ymax = max(0,object_ymax)
      im_2 = cv2.rectangle(im_1, (object_xmin,object_ymin), (object_xmax,object_ymax), (0,255,0), 3)
    else:
      im_2 = im_1

    text_1 = f"< {subject_name} , {verb_name} , {object_name} >"
    text_2 = f"< {subject_id} , {verb_id} , {object_id} >"
    org_1 = (10, 50)
    org_2 = (10, 100)
    im_3 = cv2.putText(im_2, text_1, org_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    image = cv2.putText(im_2, text_2, org_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv.putText(	img, text, Point org , int fontFace, double fontScale, color , thickness , lineType, [ bottomLeftOrigin]	)
    
    filename = address.replace("/", "_")
    cv2.imwrite(output_directory + filename, image)
