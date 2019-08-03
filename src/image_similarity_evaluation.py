#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2019
@author: Akihiro Inui
"""
# TODO: more error handling
# Import libraries
import os
import turicreate as tc
from src.common.config_reader import ConfigReader


class ModelLoadingError(Exception):
    pass


class DataLoadingError(Exception):
    pass


class ImageSimilarityEvaluation:
    """
    Init, read parameters from config file
    """
    def __init__(self, setting_file: str):
        """
        Initialize class with setting config file. Process data.
        """
        assert setting_file is not None, "Give me a setting file dude"
        self.cfg = ConfigReader(setting_file)
        self.__process__data()

    def __process__data(self):
        """
        Load image files, apply pre-process and save as sframe
        """
        # Load image data from directory
        data_process_error = []
        self.train_data = tc.image_analysis.load_images(self.cfg.train_directory)
        self.train_data = self.train_data.add_row_number()
        self.train_data.save(os.path.join(self.cfg.train_directory, '../train.sframe'))

        #data_process_error.append("Train data not loaded")
        self.test_data = tc.image_analysis.load_images(self.cfg.test_directory)
        self.test_data = self.test_data.add_row_number()
        self.test_data.save(os.path.join(self.cfg.test_directory, '../test.sframe'))

    def train(self, output_model_path: str = None):
        """
        Train model
        :param: output_model_path: Model file path to be saved with .model format
        """
        # Train model with ResNet pre-trained model
        trained_model = tc.image_similarity.create(self.train_data)
        # Save model
        if output_model_path:
            trained_model.save(output_model_path)
        return trained_model

    def evaluation(self, image_file_path: str, model=None, model_file_path: str = None):
        """
        :param:  image_file_path: Path to input image file
        :param:  model: Model to be passed directly
        :param:  model_file_path: Model file path to be loaded
        :return: similar image_path: k number of similar image file paths
        :rtype:  similar image_path: sframe
        """
        # Error
        evaluation_error = []

        # Load model if it is not passed directly
        if not model:
            try:
                model = tc.load_model(model_file_path)
                if not model.model:
                    raise ModelLoadingError
            except ModelLoadingError:
                evaluation_error.append("Model file was not passed properly")

        # Save tmp image in sframe
        tmp_data = tc.image_analysis.load_images(self.cfg.tmp_directory)
        tmp_data = tmp_data.add_row_number()
        tmp_data.save(os.path.join(self.cfg.tmp_directory, '../tmp.sframe'))

        # Get image file ID from image file path
        # image_id = self.test_data[self.test_data['path'] == image_file_path]['id'][0]  # Leave this line for test
        image_id = tmp_data[tmp_data['path'] == os.path.join(self.cfg.tmp_directory, image_file_path)]['id'][0]

        # Query with test data (k is number of the nearest neighbors to return)
        # query_results = model.query(self.test_data[image_id]['image'], k=self.cfg.k)   # Leave this line for test
        query_results = model.query(tmp_data[image_id]['image'], k=self.cfg.k)

        # Visualize result
        # self.test_data[image_id]['image'].show()  # Leave this line for test
        # tmp_data[image_id]['image'].show()  # Leave it for debug

        # Show result
        similar_image_ids = query_results[query_results['query_label'] == 0]['reference_label']
        similar_image_paths = self.train_data.filter_by(similar_image_ids, 'id')['path']
        # self.train_data.filter_by(similar_image_ids, 'id').explore()
        return similar_image_paths


def main():
    # Set condition
    train = True

    # Initialize Image Similarity class
    ISE = ImageSimilarityEvaluation(setting_file='config/master_config.ini')

    # Train model
    if train is True:
        model = ISE.train(output_model_path='model/inada.model')
        similar_image_paths = ISE.evaluation(image_file_path='dataset/test/inada.jpg', model=model)
    else:
        # Evaluate target image file with pre-trained model
        similar_image_paths = ISE.evaluation(image_file_path='dataset/test/inada.jpg', model_file_path='model/inada.model')


if __name__ == "__main__":
    main()
