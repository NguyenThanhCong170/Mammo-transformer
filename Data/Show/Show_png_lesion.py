import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd

study_id = 'eaffca1d93f249b54f4dfb6620060fb5'
image_id = '87f492b1ef0b4090691311a40b81da6f'
img_path = 'images_png' + '/eaffca1d93f249b54f4dfb6620060fb5' +'/87f492b1ef0b4090691311a40b81da6f.png'

img = Image.open(img_path)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap='gray') 

df = pd.read_csv('finding_annotations.csv')
row = df[df['image_id'] == image_id]
xmin, ymin = row['xmin'].values, row['ymin'].values
xmax, ymax = row['xmax'].values, row['ymax'].values
box_width = xmax - xmin
box_height = ymax - ymin
print(xmin,ymin,xmax,ymax,box_width,box_height)

rect = patches.Rectangle((xmin, ymin), box_width, box_height, 
                        linewidth=2, edgecolor='red', facecolor='none')

ax.add_patch(rect)
plt.axis('off') 
plt.show() 