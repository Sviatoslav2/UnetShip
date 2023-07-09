import random
import os
import config
from sklearn.model_selection import train_test_split
import data_utils


class DataGenIndex:
    def __init__(self, data, batch_size=config.ConfigUtils().batch_size, number_sup_set=config.ConfigUtils().number_sup_set):
        self.data = data
        self.__max_ShipAreaPercentage = max(self.ShipAreaPercentageList())
        self.__min_ShipAreaPercentage = min(self.ShipAreaPercentageList())
        self.batch_size = batch_size
        self.list_of_id = self.list_of_names()
        self.list_of_ShipAreaPercentageList = self.ShipAreaPercentageList()
        self.number_sup_set = number_sup_set

    def list_of_names(self):
        return list(self.data['ImageId'].values)

    def ShipAreaPercentageList(self):
        return list(self.data['ShipAreaPercentage'].values)

    def get_boundary_delta(self):
        return (self.__max_ShipAreaPercentage - self.__min_ShipAreaPercentage) / self.number_sup_set

    def get_subset_indexes(self, number_sup_set_index):
        delta = self.get_boundary_delta()
        res = [self.list_of_id[self.list_of_ShipAreaPercentageList.index(area)] for area in self.list_of_ShipAreaPercentageList if (self.__min_ShipAreaPercentage + delta * number_sup_set_index < area < self.__min_ShipAreaPercentage + delta * (number_sup_set_index + 1))]
        return res

    def get_batch_indexes(self):
        res = []
        for i in range(self.batch_size):
            index = random.choice(self.get_subset_indexes(i%self.number_sup_set))
            res.append(self.get_data_by_name(index)['path'].iloc[0])
        return res

    def get_data_by_name(self, name):
        return self.data.loc[self.data["ImageId"] == name]

    def get_batch_paths(self):
        return [data_utils.get_path_to_image(index) for index in self.get_batch_indexes()]

class DataSplit:
    def __init__(self, path_to_csv, split_test=config.ConfigUtils().test_train_split):
        data = data_utils.get_data(path_to_csv)
        self.__train, self.__test = train_test_split(data, test_size=split_test)

    def get_train(self):
        return self.__train

    def get_test(self):
        return self.__test


class DataGenUtils:
    def __init__(self, path_to_data=config.ConfigPath().train_csv):
        self.data_split = DataSplit(path_to_csv=path_to_data)
        self.path_train_img = config.ConfigPath().train_img
        self.path_test_img = config.ConfigPath().test_img

    def get_index_train(self):
        return DataGenIndex(self.data_split.get_train())

    def get_index_test(self):
        return DataGenIndex(self.data_split.get_test())
