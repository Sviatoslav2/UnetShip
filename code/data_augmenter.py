import imgaug.augmenters as iaa
import config

seq = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    iaa.CoarseDropout(0.1, size_percent=0.2),
    iaa.ElasticTransformation(alpha=50, sigma=5),
    iaa.Resize(size=(config.ConfigUtils().size_augment, config.ConfigUtils().size_augment))
], random_order=True)



seq_input = iaa.Sequential([iaa.Resize(size=(config.ConfigUtils().size_augment, config.ConfigUtils().size_augment))])