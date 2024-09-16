from collections import defaultdict
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from torch import tensor as torchtensor
from torch import long as torchlong

#remove after testing
import numpy as np
import torch


class Pingjun(Dataset):
    def __init__(
        self,
        imageSize,
        root_dir="/home/davids/Documents/data/pingjun_preprocessed_no_prosthetics",
    ):  # "./../../data/pingjun_preprocessed"):
        self.root_dir = root_dir
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Grayscale(num_output_channels=1),
        #         transforms.Resize(imageSize),
        #         transforms.ToTensor(),  # try with floating points to 16 bits
        #         # transforms.Normalize((0.5,), (0.5,)),
        #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     ]
        # )

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(imageSize),
                transforms.ToTensor(),  # try with floating points to 16 bits
                transforms.RandomHorizontalFlip(p=0.5),
                # 2. Small rotation (+/- 2 degrees)
                transforms.RandomRotation(degrees=1),
                # 3. Small affine transformations (includes warping and affine)
                # transforms.RandomAffine(degrees=1, translate=(0.02, 0.02), shear=2),
                # # 4. Random Blur (Kernel size 3x3 with a probability of 0.5)
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                # # 5. Random contrast
                # transforms.ColorJitter(contrast=0.1),
                # # 6. Random brightness
                # transforms.ColorJitter(brightness=0.1),
            ]
        )

        self.image_paths = []
        self.labels = []

        # Iterate through the first-level directories
        for class_name in os.listdir(root_dir):
            if class_name in ["train", "val", "test"]:  # ["train", "val", "test"]
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    for subfolder_name in os.listdir(class_path):
                        if subfolder_name in [
                            "2",
                            "3",
                            "4",
                        ]:  # ["0", "1", "2", "3", "4"]
                            subfolder_path = os.path.join(class_path, subfolder_name)
                            if os.path.isdir(subfolder_path):
                                for image_name in os.listdir(subfolder_path):
                                    image_path = os.path.join(
                                        subfolder_path, image_name
                                    )
                                    if image_path.lower().endswith((".png")):
                                        self.image_paths.append(image_path)
                                        self.labels.append(subfolder_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label = int(self.labels[idx])

        # Ensure label is a tensor
        label = torchtensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label



def assign_label(binary_classification, kl_class_name):
    if binary_classification:
        if kl_class_name in ["0", "1"]:
            return 0
        else:
            return 1
    else:
        return int(kl_class_name)


class Pingjun_cond(Dataset):
    def __init__(
        self,
        imageSize,
        include_dataset_types=["train", "val", "test"],
        include_kl_grades = ["0", "1", "2", "3", "4"],
        binary_classification = True,
        root_dir="/home/davids/Documents/data/pingjun_preprocessed_no_prosthetics",
    ):
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(imageSize),
                transforms.ToTensor(),  # try with floating points to 16 bits
                transforms.RandomHorizontalFlip(p=0.5),
                # 2. Small rotation (+/- 2 degrees)
                transforms.RandomRotation(degrees=1),
            ]
        )

        self.image_paths = []
        self.labels = []

        # Iterate through the first-level directories
        for dataset_type in os.listdir(root_dir):
            if dataset_type in include_dataset_types:  # ["train", "val", "test"]
                dataset_path = os.path.join(root_dir, dataset_type)
                if os.path.isdir(dataset_path):
                    for kl_grade in os.listdir(dataset_path):
                        if kl_grade in include_kl_grades:  # ["0", "1", "2", "3", "4"]
                            kl_grade_path = os.path.join(dataset_path, kl_grade)
                            if os.path.isdir(kl_grade_path):
                                for image_name in os.listdir(kl_grade_path):
                                    image_path = os.path.join(kl_grade_path, image_name)
                                    if image_path.lower().endswith((".png")):
                                        self.image_paths.append(image_path)
                                        self.labels.append(
                                            assign_label(
                                                binary_classification, kl_grade
                                            )
                                        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label = int(self.labels[idx])

        # Ensure label is a tensor
        label = torchtensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label


# class Pingjun_cond_balanced(Dataset):
#     def __init__(
#         self,
#         imageSize,
#         include_dataset_types=["train", "val", "test"],
#         include_kl_grades = ["0", "1", "2", "3", "4"],
#         binary_classification = True,
#         root_dir="/home/davids/Documents/data/pingjun_preprocessed_no_prosthetics",
#     ):
#         self.root_dir = root_dir
#         self.transform = transforms.Compose(
#             [
#                 transforms.Grayscale(num_output_channels=3),
#                 transforms.Resize(imageSize),
#                 transforms.ToTensor(),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 # transforms.RandomRotation(degrees=1),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         )

#         self.image_paths = []
#         self.labels = []

#         # Iterate through the first-level directories
#         for dataset_type in os.listdir(root_dir):
#             if dataset_type in include_dataset_types:  # ["train", "val", "test"]
#                 dataset_path = os.path.join(root_dir, dataset_type)
#                 if os.path.isdir(dataset_path):
#                     for kl_grade in os.listdir(dataset_path):
#                         if kl_grade in include_kl_grades:  # ["0", "1", "2", "3", "4"]
#                             kl_grade_path = os.path.join(dataset_path, kl_grade)
#                             if os.path.isdir(kl_grade_path):
#                                 for image_name in os.listdir(kl_grade_path):
#                                     image_path = os.path.join(kl_grade_path, image_name)
#                                     if image_path.lower().endswith((".png")):
#                                         self.image_paths.append(image_path)
#                                         self.labels.append(
#                                             assign_label(
#                                                 binary_classification, kl_grade
#                                             )
#                                         )
        
#         # Count the number of samples per class
#         class_counts = defaultdict(int)
#         for _, label in self.labels:
#             class_counts[label] += 1
        
#         # Find the minimum class count
#         min_class_count = min(class_counts.values())

#         # Sample equally from each class
#         class_sample_counts = defaultdict(int)
#         for image_path, label in self.image_paths:
#             if class_sample_counts[label] < min_class_count:
#                 self.image_paths.append(image_path)
#                 self.labels.append(label)
#                 class_sample_counts[label] += 1


#         numDataPoints = 1000
#         data_dim = 5
#         bs = 100

#         target = torch.tensor([0, 1, 2, 3, 4])
#         train_dataset = torch.utils.data.TensorDataset()

#         class_sample_count = np.array(
#             [len(np.where(target == t)[0]) for t in np.unique(target)])
#         weight = 1. / class_sample_count
#         samples_weight = np.array([weight[t] for t in target])

#         samples_weight = torch.from_numpy(samples_weight)
#         samples_weigth = samples_weight.double()
#         sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

#         train_loader = DataLoader(
#             train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

       


#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path)

#         label = int(self.labels[idx])

#         # Ensure label is a tensor
#         label = torchtensor(label)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

