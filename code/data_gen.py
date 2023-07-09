import numpy as np
import tensorflow as tf
import data_utils
import data_gen_utils
import data_augmenter
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, train_key:bool):
        self.data_utils_index_train_test = data_gen_utils.DataGenUtils()
        if train_key:
            self.data_utils = self.data_utils_index_train_test.get_index_train()
        else:
            self.data_utils = self.data_utils_index_train_test.get_index_test()
        self.__paths = self.data_utils.get_batch_paths()

    def __read_image(self, name):
        return data_utils.get_image(name)

    def __get_indexes(self):
        self.__paths = self.data_utils.get_batch_paths()

    def __get_input_images(self):
        return [self.__read_image(i) for i in self.__paths] # / 255.0

    def __get_masks(self):
        return [data_utils.mask_image(self.data_utils.data, i) // 255 for i in self.__paths]

    def __batch_images_mask(self):
        self.__get_indexes()
        return self.__get_input_images(), self.__get_masks()

    def images_mask(self):
        images, masks = self.__batch_images_mask()
        return images, masks

    def augment_images_mask(self):
        lst_aug, lst_mask = [], []
        images, masks = self.images_mask()
        for i in range(len(images)):
            images_aug, masks_aug = data_augmenter.seq(image=images[i], segmentation_maps=SegmentationMapsOnImage(masks[i], shape=(images[i].shape[0], images[i].shape[1])))
            lst_aug.append(images_aug)
            mask = masks_aug.draw()[0]
            lst_mask.append(mask[:,:,0]*mask[:,:,1]*mask[:,:,2])
        return lst_aug, lst_mask


    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        X, y = self.augment_images_mask()
        X = [x / 255.0 for x in X]
        return tf.cast(X, tf.float64), tf.cast(y, tf.float64)

    def __len__(self):
        return len(self.data_utils.list_of_id) // self.data_utils.batch_size