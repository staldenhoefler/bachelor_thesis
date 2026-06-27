import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import yaml
import pickle
from PIL import Image


class BeforeAfterWithBoxImageDataset(Dataset):
    def __init__(self, params, images=None, labels=None, transform=None):
        self.transform = transform
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.main_wd = os.path.join(SCRIPT_DIR, '..', '..')
        self.images = images
        self.labels = labels
        self.params = params

        if self.images is None:
            self.load_data()
            print(f'Size in MB of self.images: {self.images.__sizeof__() / (1024 * 1024)} MB')
            self.load_additional_after_data()
            print(f'Size in MB of self.images: {self.images.__sizeof__() / (1024 * 1024)} MB')
            self.load_additional_false_images()
            print(f'Size in MB of self.images: {self.images.__sizeof__() / (1024 * 1024)} MB')

        self.combine_labels = self.params['data'].get('combine_labels', False)
        if self.params['data'].get('combine_labels', False):
            self.label_mapping = self.params['data']['combine_labels_mapping']


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

        Is used by PyTorch DataLoader to fetch a single sample from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the (image, label).
        """
        img = self.images[idx]
        label = self.labels[idx]

        if self.params['data']['add_after_image_as_feature_channel']:
            img = np.vstack((img['before'], img['after']))
        else:
            img = np.hstack((img['before'], img['after']))

        if self.transform:
            img = self.transform(img)

        if self.combine_labels:
            label = self.label_mapping.get(label, label)
            label = int(label)

        return img, label

    def load_data(self):
        """
        Loads the dataset from a YAML file and populates the images and labels lists.
        The YAML file should contain paths to images and their corresponding labels.
        """

        data_path = os.path.join(self.main_wd, self.params["data"]["path"])
        with open(f'{data_path}/train_stripes.pkl', 'rb') as f:
            self.images = pickle.load(f)

        target_height = self.params['data']['import_images_resize_size']['height']
        target_width = self.params['data']['import_images_resize_size']['width']
        labels_df = pd.read_excel(f'{data_path}/train_labels_before_after_with_box.xlsx')
        labels_list = labels_df['Rating_clean'].tolist()
        resized_images = []
        appended_labels = []
        countr = 0
        for i, img in self.images.iterrows():
            if labels_list[countr] == 0:
                countr += 1
                continue
            img_before = self.resize_image(img['before'], target_height, target_width)
            img_after = self.resize_image(img['after'], target_height, target_width)
            img['before'] = img_before
            img['after'] = img_after
            appended_labels.append(labels_list[countr])
            resized_images.append(img)
            countr += 1
        self.images = resized_images
        self.labels = appended_labels

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
        return cv2.resize(img, (target_width, target_height))

    def load_additional_after_data(self):
        '''
        Loads additional after images from the after-dataset. the before image is created as a zero image.
        Limits the number of images with label 1 to 200 for a better class balance
        :return:
        '''
        data_path = os.path.join(self.main_wd, 'data/train')
        counter = 0
        for file in os.listdir(os.path.join(data_path, 'img')):
            if file.endswith(".jpg"):

                df = pd.read_excel(f'{data_path}/labels/train_val.xlsx')
                label = df.loc[df['Image_path'] == file, 'Rating_clean'].iloc[0]

                if (label == 1 and counter <= 200) or (label != 1):
                    img_after = cv2.imread(os.path.join(os.path.join(data_path, 'img'), file))
                    if img_after is None:
                        print(f"Error loading image {file}")
                        continue
                    img_after = self.clean_image(img_after)
                    sizes = self.params['data']['import_images_resize_size']
                    img_height, img_width = sizes['height'], sizes['width']
                    img_after = self.resize_image(img_after, img_height, img_width)
                    img = {
                        'before': np.zeros((img_height, img_width, 3), dtype=np.uint8),
                        'after': img_after
                    }

                    self.images.append(img)
                    self.labels.append(label)

                    if label == 1:
                        counter += 1


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

    def load_additional_false_images(self):
        '''
        Loads additional wrong cutted images.
        These images are randomly selected from the original dataset and cutted to a random size.
        '''


        data_path = os.path.join(self.main_wd, 'data/before_after_with_box/before_after_img_raw_png')
        file_list = os.listdir(data_path)

        for i in range(self.params['data']['additional_false_images_count']):
            random_file_nr = np.random.randint(1, len(file_list)/2)

            before_path = os.path.join(data_path, f'before_{random_file_nr}.png')
            after_path = os.path.join(data_path, f'after_{random_file_nr}.png')
            image_before = Image.open(before_path)
            image_after = Image.open(after_path)
            image_before = np.array(image_before)
            image_after = np.array(image_after)

            random_x, random_y = np.random.randint(0, 2000, size=2)
            rand_w = np.random.randint(130, 750)
            rand_h = np.random.randint(130, 1500)

            strip_before = image_before[random_y:random_y + rand_h, random_x:random_x + rand_w]
            strip_after = image_after[random_y:random_y + rand_h, random_x:random_x + rand_w]

            sizes = self.params['data']['import_images_resize_size']
            img_height, img_width = sizes['height'], sizes['width']
            strip_before = self.resize_image(strip_before, img_height, img_width)
            strip_after = self.resize_image(strip_after, img_height, img_width)

            img = {
                'before': strip_before,
                'after': strip_after
            }

            self.images.append(img)
            self.labels.append(0)


