import code
import random




data_train = code.DataGen(True)
data_test = code.DataGen(False)

code.show_image_with_encoded_pixels(data_train.data_utils.data, random.choice(data_train.data_utils.get_batch_paths()))
code.show_image_with_encoded_pixels(data_test.data_utils.data, random.choice(data_test.data_utils.get_batch_paths()))

img_train, mask_train = data_train.images_mask()
code.show_from_batch(img_train[0], mask_train[0])


img_train_aug, mask_train_aug = data_train.augment_images_mask()
img_test_aug, mask_test_aug = data_test.augment_images_mask()
code.show_from_batch(img_test_aug[0], mask_test_aug[0])


