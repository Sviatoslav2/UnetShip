import os

class ConfigPath:
    def __init__(self):
        self.parent_path = os.path.dirname(os.path.abspath('__file__'))
        self.path_to_data = os.path.join(self.parent_path, "data")
        self.test_img = os.path.join(self.path_to_data, "test_v2")
        self.train_img = os.path.join(self.path_to_data, "train_v2")
        self.train_csv = os.path.join(self.path_to_data, "train_ship_segmentations_v2.csv")
        self.test_csv = os.path.join(self.path_to_data, "sample_submission_v2.csv")
        self.model_dir = os.path.join(self.parent_path, "model")
        self.path_to_model = os.path.join(self.model_dir, "model.h5")

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigPath, cls).__new__(cls)
        return cls.instance


class ConfigUtils:
    def __init__(self):
        self.size = 768
        self.channels = 3
        self.batch_size = 64
        self.number_sup_set = 4
        self.size_augment = 224
        self.number_epoch = 100
        self.test_train_split = 0.3
        self.lr = 1e-4

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigUtils, cls).__new__(cls)
        return cls.instance