import matplotlib.pyplot as plt
import numpy as np
import os
import data_utils

def show_image_with_encoded_pixels(data, path: str):
    image_id = os.path.basename(path)
    rows = data[data['ImageId'] == image_id]
    if len(rows) == 0:
        return

    image = data_utils.get_image(path)
    image_size, _, _ = image.shape
    ship_count = len(rows)
    all_ships = np.zeros_like(image)

    ax_rows_number = ship_count + 1
    f, ax = plt.subplots(ax_rows_number, 3, figsize=(15, 5 * ax_rows_number))

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

        ax[i, 0].imshow(image)
        ax[i, 1].imshow(ship)
        ax[i, 2].imshow(image * (ship // 255))

    ax[ship_count, 0].imshow(image)
    ax[ship_count, 1].imshow(all_ships)
    ax[ship_count, 2].imshow(image * (all_ships // 255))
    plt.show()


def show_from_batch(img, mask):
    image = img
    mask = mask * 225
    f, ax = plt.subplots(1, 2, figsize=(15, 5 * 1))
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()