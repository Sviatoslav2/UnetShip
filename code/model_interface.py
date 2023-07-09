import os.path
import loss_utils
import config
import data_utils
import data_augmenter
import tensorflow as tf

class ModelInterface:
    def __init__(self, path_to_model=config.ConfigPath().path_to_model):
        self.model = tf.keras.models.load_model(path_to_model, custom_objects={'dice_coef_loss': loss_utils.dice_coef_loss, 'dice_coef': loss_utils.dice_coef, 'iou':loss_utils.iou})

    def __call__(self, X):
        if X.shape == (config.ConfigUtils().batch_size, config.ConfigUtils().size_augment, config.ConfigUtils().size_augment, config.ConfigUtils().channels):
            return self.model.predict(X)
        if X.shape == (1, config.ConfigUtils().size_augment, config.ConfigUtils().size_augment, config.ConfigUtils().channels):
            return self.model.predict(X)
        if X.shape == (config.ConfigUtils().size_augment, config.ConfigUtils().size_augment, config.ConfigUtils().channels):
            return self.model.predict(tf.expand_dims(X,axis=0))

    def predict_from_image(self, path):
        if os.path.isfile(path):
            return tf.convert_to_tensor(data_augmenter.seq_input(image=data_utils.get_image(path)), dtype=tf.float32)
        path = data_utils.get_path_to_image(path)
        if path:
            return tf.convert_to_tensor(data_augmenter.seq_input(image=data_utils.get_image(path)), dtype=tf.float32)
        else:
            ValueError("path dose not exist!")

