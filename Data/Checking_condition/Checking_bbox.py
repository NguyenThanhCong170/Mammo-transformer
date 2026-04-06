import pandas as pd

df = pd.read_csv('finding_annotations.csv')

condition_xmin = (df['xmin'] < 5)
condition_ymin = (df['ymin'] < 5)
condition_xmax = (df['xmax'] > df['width'])
condition_ymax = (df['ymax'] > df['height'])

out_of_bounds_xmin = df[condition_xmin]
out_of_bounds_xmax = df[condition_xmax]
out_of_bounds_ymin = df[condition_ymin]
out_of_bounds_ymax = df[condition_ymax]

print(len(out_of_bounds_xmin),len(out_of_bounds_xmax),len(out_of_bounds_ymin),len(out_of_bounds_ymax))

# print(out_of_bounds_min[['image_id', 'xmin', 'ymin', 'width', 'height']].head())
# print(out_of_bounds_max[['image_id', 'xmax', 'ymax', 'width', 'height']].head())