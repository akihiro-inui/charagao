#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:39:31 2017
Pre process to image files
@author: Akihiro Inui
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
from utils.file_utils import FileUtil
from utils.image_util import ImageUtil
from common.config_reader import ConfigReader
from dataset.data_process import DataProcess
from dataset.torch_dataset import TorchDataset
from sklearn.model_selection import train_test_split


class ImagePreProcess:
    """
    Image pre-processing to image files
    """

    # Initialization
    def __init__(self, setting_file: str):
        # Load parameters from config file
        self.cfg = ConfigReader(setting_file)
        self.pre_process_selection_dict = self.cfg.section_reader("pre_process_selection")
        self.pre_process_list = self.__init_pre_process_select()
        DataProcess.make_torch_dataset(self.cfg.dataset_directory, self.cfg.dataset_info_file, sub_folder=True)

    def __init_pre_process_select(self) -> list:
        """
        Extract setting for pre-processing from config file
        :return list of features to extract
        """
        pre_process_list = []
        for feature, switch in self.pre_process_selection_dict.items():
            if switch == "True":
                pre_process_list.append(feature)
        return pre_process_list

    def process_file(self, input_image_file) -> dict:
        """
        Apply pre process to image file. Apply selected pre-process methods
        :return dictionary of pre-processed image data in numpy array {key:process type, value: numpy array}
        :rtype  dict
        """
        image_process_dict = {}
        for pre_process in self.pre_process_list:
            original_image_numpy_array = ImageUtil.resize(ImageUtil.read_image_file(input_image_file), self.cfg.image_size)
            # Original image
            if pre_process == "original":
                image_process_dict[pre_process] = ImageUtil.flatten_image(original_image_numpy_array)
            elif pre_process == "noise":
                image_process_dict[pre_process] = ImageUtil.flatten_image(ImageUtil.noise_image(original_image_numpy_array))
            elif pre_process == "blur":
                image_process_dict[pre_process] = ImageUtil.flatten_image(ImageUtil.blur_image(original_image_numpy_array))
            elif pre_process == "zoom_out":
                image_process_dict[pre_process] = ImageUtil.flatten_image(ImageUtil.zoom_out_image(original_image_numpy_array))
            elif pre_process == "zoom_in":
                image_process_dict[pre_process] = ImageUtil.flatten_image(ImageUtil.zoom_in_image(original_image_numpy_array))
            elif pre_process == "mirror":
                image_process_dict[pre_process] = ImageUtil.flatten_image(ImageUtil.mirror_image(original_image_numpy_array))
        return image_process_dict

    def process_directory(self, input_directory):
        """
        Apply pre process to image files in a directory. Apply selected pre-process methods
        :return dictionary of pre-processed image data in numpy array {key:file name value: dictionary {key:process type, value: numpy array}}
        :rtype  dict
        """
        # Extract file names in the input directory
        file_names = FileUtil.get_file_names(input_directory)

        # Apply pre-process to image files in a directory
        files_data_dict = {}
        start = time.time()
        for image_file in file_names:
            # Apply pre-process to one image file
            files_data_dict[image_file] = self.process_file(os.path.join(input_directory, image_file))
        end = time.time()

        print("Processed {0} with {1} \n".format(input_directory, end - start))
        return files_data_dict

    def process_dataset(self, dataset_path: str):
        """
        Apply Process to image file. Apply selected pre-process methods
        :param:  dataset_path:  path to the dataset
        :return: pre processed image data
        :rtype:  pandas dataframe
        """
        # Get folder names under data set path
        category_names = FileUtil.get_folder_names(dataset_path, sort=True)

        # Extract all features and store them into list
        final_dataframe = pd.DataFrame()
        for category in category_names:
            # Apply pre process to a directory
            files_data_dict = self.process_directory(os.path.join(dataset_path, category))

            # Convert dictionary to list
            category_data_list = DataProcess.dict2list(files_data_dict)

            # Convert list to dataframe
            category_data_dataframe = pd.DataFrame(category_data_list)

            # Add label to data frame
            class_dataframe_with_label = DataProcess.add_label(category_data_dataframe, category)

            # Combine data frames
            final_dataframe = final_dataframe.append(class_dataframe_with_label)
        return final_dataframe

    @staticmethod
    def reshape_dataframe(input_dataframe, image_size: int, num_colors: int):
        """
        Square image, flatten it and normalize.
        :type   input_dataframe: input data in pandas dataframe
        :param  input_dataframe: input data in pandas dataframe
        :param  image_size: image size
        :param  num_colors: number of colors
        :rtype  numpy array
        :return reshaped numpy array
        """
        # Convert to numpy array
        numpy_array = np.asarray(input_dataframe)
        # Reformat to multi-dimensional array
        return numpy_array.reshape(numpy_array.shape[0], image_size, image_size, num_colors)

    @staticmethod
    def torch_train_test(dataset_root_directory: str, image_size: int, dataset_info_csv_file: str, test_rate: float):
        """
        Using torchvision Dataset class, this method will create train and test dataset
        :param dataset_root_directory: Root directory of dataset
        :param image_size: Image size to resize
        :param dataset_info_csv_file: csv where file names and labels are written
        :param test_rate: Rate for test data (~0.3)
        """
        # Read csv file and make custom dataset for Torch
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(80, resample=False, expand=False, center=None),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5], [0.5])
        ])

        # Torch Dataset
        Dataset = TorchDataset(dataset_root_directory, dataset_info_csv_file, transform=transform)

        # Split dataset into data and label
        train_data_with_label, test_data_with_label = train_test_split(Dataset, test_size=test_rate, stratify=Dataset.image_dataframe['label'])

        return train_data_with_label, test_data_with_label

    @staticmethod
    def torch_data_loader(train_data_with_label, test_data_with_label, validation_rate: float):
        """
        Using torchvision Dataset class, this method will create train and test dataset loader
        :param train_data_with_label: Training data with label
        :param test_data_with_label:  Test data with label
        :param validation_rate: Rate for validation dataset
        """
        # Split training data into train adn validation
        train_data_with_label, validation_data_with_label = train_test_split(train_data_with_label, test_size=validation_rate)

        # Make data loader with batch
        train_loader = torch.utils.data.DataLoader(train_data_with_label, batch_size=int(len(train_data_with_label)/5)+1, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_data_with_label, batch_size=int(len(validation_data_with_label)/3)+1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data_with_label, batch_size=int(len(test_data_with_label)/3)+1, shuffle=False)
        return train_loader, validation_loader, test_loader

    def process_image(image_path, img_size):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch
        model, returns an Numpy array
        :param image_path: Input image file path
        """
        # Open the image
        from PIL import Image
        img = Image.open(image_path)

        # Resize
        if img.size[0] > img.size[1]:
            img.thumbnail((10000, img_size))
        else:
            img.thumbnail((img_size, 10000))

        # Crop
        left_margin = (img.width-224)/2
        bottom_margin = (img.height-224)/2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        img = img.crop((left_margin, bottom_margin, right_margin,
                        top_margin))
        # Normalize
        img = np.array(img)/255
        mean = np.array([0.485, 0.456, 0.406])  # provided mean
        std = np.array([0.229, 0.224, 0.225])   # provided std
        img = (img - mean)/std
    
        # Move color channels to first dimension as expected by PyTorch
        img = img.transpose((2, 0, 1))
    
        return img