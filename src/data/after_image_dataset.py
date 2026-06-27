import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import yaml


class AfterImageDataset(Dataset):
    """
        A PyTorch Dataset for loading afterimages with their corresponding labels.

        This dataset can be initialized in two ways:
        1. By providing preloaded lists of images and labels.
        2. By letting the class load the data from a directory structure
           specified in a `params.yaml` file.

        The dataset automatically handles cleaning (removing borders), resizing,
        and optional transformations.

        Attributes:
            data_path (str): The base path to the dataset directory.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the images.
            images (list): A list of loaded and preprocessed image data.
            labels (list): A list of corresponding labels for the images.
        """

    def __init__(self, params, images: list = [], labels: list = [], train: bool = True, transform: callable = None):
        """
        Initializes the AfterImageDataset.

        Args:
            images (list, optional): A list of preloaded images (e.g., numpy arrays).
                                     If empty, data will be loaded from disk. Defaults to [].
            labels (list, optional): A list of preloaded labels corresponding to the images.
                                     Defaults to [].
            train (bool, optional): Specifies whether to load the training or testing dataset.
                                    This affects the data path and label file used. Defaults to True.
            transform (callable, optional): A PyTorch transform to be applied to each image
                                            on-the-fly. Defaults to None.

        Usage:
            # Load data from files specified in params.yaml
            dataset = AfterImageDataset(train=True, transform=some_transform)

            # Use preloaded data
            dataset = AfterImageDataset(images=my_images, labels=my_labels, transform=some_transform)
        """
        self.data_path = ''
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = labels
        self.label_mapping = {}
        self.params = params

        self.combine_labels = self.params['data'].get('combine_labels', False)

        if self.combine_labels:
            self.label_mapping = self.params['data']['combine_labels_mapping']

        if len(images) == 0:
            self.load_data()
        else:
            self.images = images
            self.labels = labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        Is used by PyTorch DataLoader to determine the size of the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves an image and its corresponding label from the dataset at the specified index.

        The label is adjusted to be zero-indexed (original_label - 1). If a transform
        is provided, it is applied to the image.

        Is used by PyTorch DataLoader to fetch a single sample from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the (image, label).
        """
        img = self.images[idx]

        label = self.labels[idx][0]

        if self.transform:
            img = self.transform(img)

        if self.combine_labels:
            label = self.label_mapping.get(label, label)
            label = int(label)

        return img, label

    def extract_label(self, filename: str) -> any:
        """
        Extracts the label for a given image filename from an Excel file.

        The method reads from 'train_val.xlsx' or 'test.xlsx' based on the
        `self.train` attribute.

        Args:
            filename (str): The name of the image file (e.g., 'image_001.jpg').

        Returns:
            any: The label corresponding to the filename.
        """
        if self.train:
            df = pd.read_excel(f'{self.data_path}/labels/train_val.xlsx')
        else:
            df = pd.read_excel(f'{self.data_path}/labels/test.xlsx')

        label_column = self.params['data'].get('label_column', "Rating_clean")

        label = df[df['Image_path'] == filename][label_column].values

        return label

    def clean_image(self, img: 'numpy.ndarray') -> 'numpy.ndarray':
        """
        Processes an image to remove black/white borders and ensures vertical orientation.

        The cleaning process involves:
        1. Converting the image to grayscale.
        2. Creating a binary mask to identify the non-border content.
        3. Finding the largest contour to define the image area.
        4. Cropping the image to this area.
        5. Rotating the image 90 degrees counter-clockwise if it's in landscape orientation.

        The process is better described in 'notebooks/00_eda_images.ipynb'.

        Args:
            img (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: The cleaned image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 1, 254)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped_image = img[y:y + h, x:x + w]

        if cropped_image.shape[1] > cropped_image.shape[0]:
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return cropped_image

    def resize_image(self, img: 'numpy.ndarray', target_height: int, target_width: int) -> 'numpy.ndarray':
        """
        Resizes an image to the specified target dimensions.

        Args:
            img (numpy.ndarray): The image to resize.
            target_height (int): The target height in pixels.
            target_width (int): The target width in pixels.

        Returns:
            numpy.ndarray: The resized image.
        """
        if self.params['data']['pad_images']:
            h, w, _ = img.shape
            max_side = max(w, h)
            pad_h = max_side - h
            pad_w = max_side - w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return cv2.resize(img, (target_width, target_height))

    def load_data(self):
        """
        Loads all image and label data from the file system.

        This method is called by `__init__` if no preloaded images are provided.
        It constructs the data path based on `params.yaml` and the `train` flag,
        then iterates through image files, loading, cleaning, resizing, and
        appending them and their corresponding labels to the dataset.

        Args:
            params (dict): A dictionary loaded from the `params.yaml` configuration file.
        """
        self.data_path = self.params['data']['path_before']
        main_wd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        self.data_path = os.path.join(main_wd, self.data_path)

        if self.train:
            self.data_path = os.path.join(self.data_path, 'train')
        else:
            self.data_path = os.path.join(self.data_path, 'test')

        for file in os.listdir(os.path.join(self.data_path, 'img')):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(os.path.join(self.data_path, 'img'), file))
                if img is None:
                    print(f"Error loading image {file}")
                    continue
                img = self.clean_image(img)
                sizes = self.params['data']['import_images_resize_size']
                img_height, img_width = sizes['height'], sizes['width']
                img = self.resize_image(img, img_height, img_width)

                label = self.extract_label(file)

                self.images.append(img)
                self.labels.append(label)
    '''''
        data_path_before_after = os.path.join(main_wd, self.params["data"]["path"])
        with open(f'{data_path_before_after}/train_stripes.pkl', 'rb') as f:
            new_after_images = pickle.load(f)

        labels_df = pd.read_excel(f'{data_path_before_after}/train_labels_before_after_with_box.xlsx')
        labels_df = labels_df.reset_index(drop=True)

        target_height = self.params['data']['import_images_resize_size']['height']
        target_width = self.params['data']['import_images_resize_size']['width']

        resized_images = []
        filtered_labels = []
        countr = 0
        for i, img in new_after_images.iterrows():
            if labels_df[labels_df.index == countr]['Rating_clean'].values[0] == 0:
                countr += 1
                continue
            else:
                img_after = self.resize_image(img['after'], target_height, target_width)
                resized_images.append(img_after)
                label = np.array(labels_df[labels_df.index == countr]['Rating_clean'])
                filtered_labels.append(label)
                countr += 1
        print(f"Loaded {len(resized_images)} after images with labels.")
        self.images = self.images + resized_images
        self.labels = self.labels + filtered_labels'''

    def _level_out_labels(self):
        def list_contains_n_labels(label_list, label, n):
            """
            Checks if the list contains at least n datapoints of the label.
            """
            return label_list.count(label) >= n

        if self.params['data']['additional_rating_label']['balanced']:
            label_counts = pd.Series(self.labels).value_counts()
            sorted_list = label_counts.sort_values(ascending=False)
            second_most_label_count = sorted_list.iloc[1]

            # find the most common label
            most_common_label = label_counts.idxmax()
            # remove n most common label so that there are the same number of samples like for the second most common label

            balanced_labels = []
            balanced_images = []
            for i in range(len(self.labels)):

                if self.labels[i] == most_common_label and list_contains_n_labels(balanced_labels, self.labels[i], second_most_label_count):
                    continue
                else:
                    balanced_labels.append(self.labels[i])
                    balanced_images.append(self.images[i])

            self.images = balanced_images
            self.labels = balanced_labels