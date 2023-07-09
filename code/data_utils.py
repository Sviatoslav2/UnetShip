import numpy as np
import pandas as pd
import cv2
import config
import os
import dask.dataframe as dd


def get_path_to_image(index):
    if os.path.isfile(os.path.join(config.ConfigPath().train_img, index)):
        return os.path.join(config.ConfigPath().train_img, index)
    elif os.path.isfile(os.path.join(config.ConfigPath().test_img, index)):
        return os.path.join(config.ConfigPath().test_img, index)

def get_train_image(name: str):
    path = os.path.join(config.ConfigPath().train_img, name)
    return cv2.imread(path)

def get_image(path: str):
    return cv2.imread(path)

def get_test_image(name: str):
    path = os.path.join(config.ConfigPath().test_img, name)
    return cv2.imread(path)

def extract_features_from_image(row):
    row['ImageHeight'], row['ImageWidth'] = config.ConfigUtils().size, config.ConfigUtils().size
    return row

def pixels_number(encoded_pixels: str) -> int:
    if pd.isna(encoded_pixels):
        return 0
    return np.array(encoded_pixels.split()[1::2], dtype=int).sum()

def get_data(path):
    dtypes = {"ImageId": "object",
                 "EncodedPixels": "object",
                 "ImageHeight": "int64",
                 "ImageWidth": "int64",
                 "ShipAreaPercentage": "float64"
                 }
    data = dd.read_csv(path, dtype=dtypes).compute()
    data['EncodedPixels'] = data['EncodedPixels'].astype('string')
    data = data.apply(lambda x: extract_features_from_image(x), axis=1)
    data['path'] = data.ImageId.apply(lambda x: get_path_to_image(x))
    data['ShipAreaPercentage'] = data.apply(lambda x: pixels_number(x['EncodedPixels']) / (x['ImageHeight'] * x['ImageWidth']) * 100, axis=1)
    corrupted_images = ['6384c3e78.jpg']
    data = data.drop(data[data['ImageId'].isin(corrupted_images)].index)
    data = data.dropna()
    return data

def mask_image(data, path):
    image_id = os.path.basename(path)
    rows = data[data['ImageId'] == image_id]
    if len(rows) == 0:
        return

    image = get_image(path)
    image_size, _, _ = image.shape
    ship_count = len(rows)
    all_ships = np.zeros_like(image, dtype=np.int32)

    for i in range(ship_count):
        image_info = rows.iloc[i]

        encoded_pixels = np.array(image_info['EncodedPixels'].split(), dtype=int)
        pixels, shift = encoded_pixels[::2], encoded_pixels[1::2]
        ship = np.zeros_like(image)

        for pixel, shift in zip(pixels, shift):
            for j in range(shift):
                cur_pixel = pixel + j - 1
                ship[cur_pixel % image_size, cur_pixel // image_size] = [255, 255, 255]
        all_ships += ship
    return all_ships

