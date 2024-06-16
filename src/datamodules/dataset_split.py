from sklearn.model_selection import train_test_split


class DatasetSplits(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_split(images, train_size=0.6):
        train_images, valtest_images = train_test_split(images, train_size=train_size)
        val_images, test_images = train_test_split(valtest_images, train_size=0.5)
        return train_images, val_images, test_images
