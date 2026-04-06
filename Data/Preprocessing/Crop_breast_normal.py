import os
import sys

import os
import dicomsdl
import numpy as np
import pandas as pd

import cv2

'''
This is code which is taken and adjusted from MAMMO-CLIP preprocessing data
This is for 20486 - 2254 last rows only in finding_annotations.csv because they don't contain bbox
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

def ExtractBreast(img):
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
    return img_copy[row_ind][:, col_ind]

def save_imgs(in_path,SAVE_FOLDER,patient_id,img_id, SIZE=(960, 2400)):
    dicom = dicomsdl.open(in_path)
    data = dicom.pixelData()
    if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    extracted_breast = ExtractBreast(data)

    #save image
    patient_folder = os.path.join(SAVE_FOLDER, str(patient_id))
    os.makedirs(patient_folder, exist_ok=True)
    save_path = os.path.join(patient_folder, f"{img_id}.png")
    img = cv2.resize(extracted_breast, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, img)

    return extracted_breast.shape[0],extracted_breast.shape[1]

df = pd.read_csv('finding_annotations.csv')
df = df.iloc[2254:]
df = df.reset_index(drop=True) 
print(df)

_SIZE = (960, 2400)
IMG_PATH = 'images'
SAVE_FOLDER = 'images_cropped'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

total_height = 0
total_width = 0
total_image = 0

for index, row in df.iterrows():
    study_id = row["study_id"]
    image_id = row["image_id"]
    _in_path = os.path.join(IMG_PATH, study_id, f"{image_id}.dicom")
    print(f"=====================>>>>> {index} <<<<<=====================")
    cropped_height, cropped_width = save_imgs(
        in_path=_in_path, SAVE_FOLDER=SAVE_FOLDER,patient_id = study_id, img_id=image_id, SIZE=_SIZE
    )
    total_height+= cropped_height
    total_width+= cropped_width
    total_image += 1
    print(f'Tỉ lệ height/width: {total_height/total_width}')
    print(f'Trung bình height: {total_height/total_image}')
    print(f'Trung bình width: {total_width/total_image}')

df['resized_xmin'] = np.nan
df['resized_ymin'] = np.nan
df['resized_xmax'] = np.nan
df['resized_ymax'] = np.nan

df.to_csv(
    'vindr_detection2.csv',
    index=False
)