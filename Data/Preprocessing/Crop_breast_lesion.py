import os
import sys

import os
import dicomsdl
import numpy as np
import pandas as pd

import cv2

'''
This is code which is taken and adjusted from MAMMO-CLIP preprocessing data
This is for 2254 first rows only in finding_annotations.csv because they contain bbox
'''

def np_CountUpContinuingOnes(b_arr):
    # indice continuing zeros from left side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    # from right side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1

def adjust_bounding_box(original_coords, left_crop, top_crop):
    x1, y1, x2, y2 = original_coords

    x1_new = x1 - left_crop
    y1_new = y1 - top_crop
    x2_new = x2 - left_crop
    y2_new = y2 - top_crop

    return x1_new, y1_new, x2_new, y2_new

def ExtractBreast(img, true_bounding_box):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, _ = img.shape
    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.47)
    x_b = width // 2 - int(width * 0.47)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]
    adjusted_coords = adjust_bounding_box(true_bounding_box, col_ind[0], row_ind[0])
    return img_copy[row_ind][:, col_ind], adjusted_coords

def save_imgs(in_path, original_bbox,SAVE_FOLDER,patient_id,img_id, SIZE=(960, 2400)):
    dicom = dicomsdl.open(in_path)
    data = dicom.pixelData()
    if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    adjusted_bbox = (
        max(0, original_bbox[0]),  # Adjust top, making sure it's not less than 0
        max(0, original_bbox[1]),  # Adjust left, making sure it's not less than 0
        min(data.shape[1], original_bbox[2]),  # Adjust bottom, considering the new image shape
        min(data.shape[0], original_bbox[3]),  # Adjust right, considering the new image shape
    )
    print("Trước khi crop (bbox): ", end = "")
    xmin,ymin,xmax,ymax = adjusted_bbox[0],adjusted_bbox[1],adjusted_bbox[2],adjusted_bbox[3]
    print(xmin,ymin,xmax-xmin,ymax-ymin)

    extracted_breast, adjusted_boxes = ExtractBreast(data, adjusted_bbox)
    resized_xmin = max(0,adjusted_boxes[0])
    resized_ymin = max(0,adjusted_boxes[1])
    resized_xmax = min(extracted_breast.shape[1],adjusted_boxes[2])
    resized_ymax = min(extracted_breast.shape[0],adjusted_boxes[3])
    resized_width = resized_xmax - resized_xmin
    resized_height = resized_ymax - resized_ymin

    print("Sau khi crop (bbox): ", end = "")
    print(resized_xmin, resized_ymin, resized_width, resized_height)

    scale_x = SIZE[0] / extracted_breast.shape[1]
    scale_y = SIZE[1] / extracted_breast.shape[0]
    resized_xmin = (resized_xmin * scale_x)
    resized_ymin = (resized_ymin * scale_y)
    resized_xmax = (resized_xmax * scale_x)
    resized_ymax = (resized_ymax * scale_y)

    # Create a Rectangle patch for the resized bounding box
    resized_width = resized_xmax - resized_xmin
    resized_height = resized_ymax - resized_ymin

    print("Sau khi resize (bbox): ", end = "")
    print(resized_xmin, resized_ymin, resized_width, resized_height)

    #save image
    patient_folder = os.path.join(SAVE_FOLDER, str(patient_id))
    os.makedirs(patient_folder, exist_ok=True)
    save_path = os.path.join(patient_folder, f"{img_id}.png")
    img = cv2.resize(extracted_breast, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, img)

    return resized_xmin, resized_ymin, resized_xmax, resized_ymax, extracted_breast.shape[0],extracted_breast.shape[1]

df = pd.read_csv('finding_annotations.csv')
df = df.head(2254)
df = df.reset_index(drop=True)

_SIZE = (960, 2400)
IMG_PATH = 'images'
SAVE_FOLDER = 'images_cropped'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
resized_xmin_arr = []
resized_ymin_arr = []
resized_xmax_arr = []
resized_ymax_arr = []

total_height = 0
total_width = 0
total_image = 0

for index, row in df.iterrows():
    study_id = row["study_id"]
    image_id = row["image_id"]
    original_bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    _in_path = os.path.join(IMG_PATH, study_id, f"{image_id}.dicom")
    print(f"=====================>>>>> {index} <<<<<=====================")
    resized_xmin, resized_ymin, resized_xmax, resized_ymax, cropped_height, cropped_width = save_imgs(
        in_path=_in_path, original_bbox=original_bbox, SAVE_FOLDER=SAVE_FOLDER,patient_id = study_id, img_id=image_id, SIZE=_SIZE
    )
    total_height+= cropped_height
    total_width+= cropped_width
    total_image += 1
    print(f'Tỉ lệ height/width: {total_height/total_width}')
    print(f'Trung bình height: {total_height/total_image}')
    print(f'Trung bình width: {total_width/total_image}')

    resized_xmin_arr.append(resized_xmin)
    resized_ymin_arr.append(resized_ymin)
    resized_xmax_arr.append(resized_xmax)
    resized_ymax_arr.append(resized_ymax)
df['resized_xmin'] = resized_xmin_arr
df['resized_ymin'] = resized_ymin_arr
df['resized_xmax'] = resized_xmax_arr
df['resized_ymax'] = resized_ymax_arr
df.to_csv(
    'vindr_detection1.csv',
    index=False
)