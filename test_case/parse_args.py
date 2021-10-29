
class Dataset1():
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self, index):
        return self.imgs[index,:] , self.labels[index]
