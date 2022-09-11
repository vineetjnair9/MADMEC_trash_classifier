#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

#%%
personal = 1
if personal:
    data_path = '/Users/vinee/Dropbox (MIT)/New House Waste Experiment/Combined_raw'

data_dir = pathlib.Path(data_path)

#%% Rename files to remove raw _R
os.chdir(data_dir)
for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    names = f_name.split('_')
    f_name = names[0] + '_' + names[1]
 
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)

#%% Count images
image_count = len(list(data_dir.glob('*.jpg'))) + len(list(data_dir.glob('*.HEIC')))
print('No. of images: ', image_count)

# %%
batch_size = 32
img_height = 384
img_width = 512

# Split into training and validation sets
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels=None,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

vals_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels=None,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
