import os
import glob
import numpy as np
from skimage.io import imread
import random
import tensorflow as tf


class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
            test_batch_size, height_of_image, width_of_image, num_channels, num_classes):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes=num_classes
        self.height_of_image=height_of_image
        self.width_of_image=width_of_image

    def load_image(self, path):
        image=imread(path)
        image=image.reshape(self.width_of_image,self.height_of_image)
        image=np.array(image)/255
        target_vector=int(os.path.basename(os.path.dirname(path)))

        label=np.eye(self.num_classes)[int(target_vector)]
        return image , label
        print(label)
        

    def batch_data_loader(self, batch_size, file_paths, index):
        images=[]
        labels=[]
        for i in range(int(index*batch_size),int((index+1)*batch_size)):
            image,label=self.load_image(file_paths[i])
            images.append(image)
            labels.append(label)

        return images , labels

    def on_epoch_end(self):
        np.random.shuffle(self.train_paths)

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)



