from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective
from basic.transforms import SignFlipping, DPIAdjusting, Dilation, Erosion, ElasticDistortion, RandomTransform
from basic.utils import LM_str_to_ind
import os
import numpy as np
import pickle
from PIL import Image
import cv2
import copy
import torch


class DatasetManager:

    def __init__(self, params):
        self.params = params
        self.img_padding_value = params["config"]["padding_value"]
        self.dataset_class = params["dataset_class"]
        self.tokens = {
            "pad": params["config"]["padding_token"],
        }
        self.train_dataset = None
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_loader = None
        self.valid_loaders = dict()
        self.test_loaders = dict()

        self.train_sampler = None
        self.valid_samplers = dict()
        self.test_samplers = dict()

        self.charset = self.get_merged_charsets()
        self.load_datasets()

        if params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = len(self.charset)
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 1
            params["config"]["padding_token"] = self.tokens["pad"]

        elif params["config"]["charset_mode"] == "attention":
            self.tokens["end"] = len(self.charset)
            self.tokens["start"] = len(self.charset) + 1
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 2
            if params["config"]["pad_label_with_end_token"]:
                params["config"]["padding_token"] = self.tokens["end"]
            else:
                params["config"]["padding_token"] = self.tokens["pad"]
        self.update_charset()
        self.my_collate_function = OCRCollateFunction(self.params["config"])

        self.load_ddp_samplers()
        self.load_dataloaders()

    def get_merged_charsets(self):
        datasets = self.params["datasets"]
        charset = set()
        for key in datasets.keys():
            with open(os.path.join(datasets[key], "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                charset = charset.union(set(info["charset"]))
        if "\n" in charset:
            charset.remove("\n")
        if "¬" in charset:
            charset.remove("¬")
        if "" in charset:
            charset.remove("")
        return sorted(list(charset))



