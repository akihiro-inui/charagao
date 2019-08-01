#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
@author: Akihiro Inui
"""

import configparser
from src.utils.file_utils import FileUtil


class ConfigReader:
    """
    Reading configuration file data. Define module specific configuration in different functions.
    """

    def __init__(self, i_config_path: str):
        """
        Read common configuration data from configuration file
        """
        cfg = configparser.ConfigParser()
        self.cfg = cfg
        config_file = i_config_path
        assert FileUtil.is_valid_file(config_file), config_file + " is not a valid configuration file!"
        cfg.read(config_file)

        # Read module specific config reader
        self.__init__dataset(cfg)
        self.__init__classifier_selection(cfg)
        self.__init__classifier_parameter(cfg)

    def __init__dataset(self, cfg):
        # Parameters for data set creation
        self.train_directory = str(cfg.get('dataset', 'train_directory'))
        self.test_directory = str(cfg.get('dataset', 'test_directory'))
        self.tmp_directory = str(cfg.get('dataset', 'tmp_directory'))

    def __init__classifier_selection(self, cfg):
        # Parameters for classifier selection
        self.ResNet = bool(cfg.get('classifier_selection', 'resnet'))

    def __init__classifier_parameter(self, cfg):
        # Parameters for classifier selection
        self.k = int(cfg.get('classifier_parameter', 'k'))

    def section_reader(self, section_name: str) -> dict:
        # Read parameters from a given section
        param_dict = {}
        options = self.cfg.options(section_name)
        for option in options:
            try:
                param_dict[option] = self.cfg.get(section_name, option)
                if param_dict[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                param_dict[option] = None
        return param_dict
