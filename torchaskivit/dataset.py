import os
import glob
import json
import torch
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


class ASKIVIT_V1(Dataset):
    """
    Alternative loader for ASKIVIT V1.3 dataset. Gets label not from json file,
    but from filename of .npy file.
    """
    def __init__(self, directory: str, class_to_idx: dict, transform=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V1, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform

        # List all .npy files in the directory / ASKIVIT dataset.
        self.filepaths_list = glob.glob(os.path.join(directory, '*.npy'))

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # Get label from filename. Classification not pixel-wise.
        npy_filename_sliced = npy_filename.split("_")[-2:]
        label = "_".join(npy_filename_sliced).split(".")[0]

        # Map label to two main classes.
        if label.split("_")[0] == "Holz":
            main_class_idx = 0
        elif label.split("_")[0] == "Nicht-Holz":
            main_class_idx = 1
        else:
            raise ValueError(f"Label {label} not in class_to_idx")
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label
    

class ASKIVIT_V1_5(Dataset):
    def __init__(self, directory: str, class_to_idx: dict, transform=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V1_5, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform

        # List all .npy files in the directory / ASKIVIT dataset.
        self.filepaths_list = glob.glob(os.path.join(directory, '*.npy'))

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # get label out of file name (true_label = mainclass, gt_label = subclass)
        npy_filename_sliced = "_".join(npy_filename.split(".")[0].split("_")[5:])

        if "Non" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:2])

            gt_label = "_".join(npy_filename_sliced.split("_")[2:])

        elif "Wood" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:1])

            gt_label = "_".join(npy_filename_sliced.split("_")[1:])

        # Map label to two main classes.
        if true_label == "Wood":
            main_class_idx = 0
        elif true_label == "Non_wood":
            main_class_idx = 1
        else:
            raise ValueError(f"Label {true_label} not in class_to_idx")
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label
    
class ASKIVIT_V1_5_NO_METAL(Dataset):
    def __init__(self, directory: str, class_to_idx: dict, transform=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V1_5_NO_METAL, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform

        # List all .npy files in the directory / ASKIVIT dataset.
        all_files = glob.glob(os.path.join(directory, '*.npy'))

        # filter all files labled with the word Metal in them
        self.filepaths_list = [file for file in all_files if "Metal" not in os.path.basename(file)]

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # get label out of file name (true_label = mainclass, gt_label = subclass)
        npy_filename_sliced = "_".join(npy_filename.split(".")[0].split("_")[5:])

        if "Non" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:2])

            gt_label = "_".join(npy_filename_sliced.split("_")[2:])

        elif "Wood" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:1])

            gt_label = "_".join(npy_filename_sliced.split("_")[1:])

        # Map label to two main classes.
        if true_label == "Wood":
            main_class_idx = 0
        elif true_label == "Non_wood":
            main_class_idx = 1
        else:
            raise ValueError(f"Label {true_label} not in class_to_idx")
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label
    
class ASKIVIT_V1_5_metal_expert(Dataset):
    def __init__(self, directory: str, class_to_idx: dict, transform=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V1_5_metal_expert, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform

        # List all .npy files in the directory / ASKIVIT dataset.
        self.filepaths_list = glob.glob(os.path.join(directory, '*.npy'))

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # get label out of file name (true_label = mainclass, gt_label = subclass)
        npy_filename_sliced = "_".join(npy_filename.split(".")[0].split("_")[5:])

        # wood mentioned because of main label cases (wood vs non_wood -> one more "_" for seperation)
        if "Non" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:2])

            gt_label = "_".join(npy_filename_sliced.split("_")[2:])

        elif "Wood" in npy_filename_sliced:
            true_label = "_".join(npy_filename_sliced.split("_")[0:1])

            gt_label = "_".join(npy_filename_sliced.split("_")[1:])

        # check subclass for metal and give mainclass based on that
        if "Metal" in gt_label:
            main_class_idx = 0
        else:
            main_class_idx = 1
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label

class ASKIVIT_V2_old(Dataset):
    def __init__(self, directory, sensor: str, mode="train", transform=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        sensor : str
            "RGB" or "NIR", or other in the future.
        mode : str
            "train", "val" or "test".
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V2_old, self).__init__()
        self.directory = directory
        self.transform = transform

        # List all .npy files in the directory / ASKIVIT dataset.
        pattern = os.path.join(directory, f'*/{mode}/*{sensor}.npy')
        self.files = glob.glob(pattern)

        total_size = sum(os.path.getsize(file) for file in self.files)
        print(f"Size of ASKIVIT V2.0 {mode} set {sensor}: {total_size / 1e9:.2f} GB")

        
    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        filepath = self.files[idx]

        filename = os.path.basename(filepath)
        label = "_".join(filename.split("_")[-3:-1])

        # Map label to two main classes.
        main_label = label.split("_")[0]
        if main_label == "Holz":
            main_class_idx = 0
        elif main_label == "Nicht Holz" or main_label == "Background":
            main_class_idx = 1
        else:
            raise ValueError(f"Label {label} not in class_to_idx")
        

        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(filepath, mmap_mode='r')


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label    

class ASKIVIT_V2(Dataset):
    def __init__(self, directory: str, class_to_idx: dict, transform=None, modality=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V2, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.modality = modality

        # List all .npy files in the directory / ASKIVIT dataset.
        self.filepaths_list = glob.glob(os.path.join(directory, f'{modality}_*.npy'))

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # get label out of file name
        if "GT_" not in npy_filename:

            npy_filename_sliced = "_".join(npy_filename.split(".")[1].split("_")[3:])

            if "Non_wood_" in npy_filename_sliced:
                main_label = "_".join(npy_filename_sliced.split("_")[0:2])

                sub_label = "_".join(npy_filename_sliced.split("_")[2:])

            elif "Wood_" in npy_filename_sliced:
                main_label = "_".join(npy_filename_sliced.split("_")[0:1])

                sub_label = "_".join(npy_filename_sliced.split("_")[1:])

        # Map label to two main classes.
        if main_label == "Wood":
            main_class_idx = 0
        elif main_label == "Non_wood":
            main_class_idx = 1
        else:
            raise ValueError(f"Label {main_label} not in class_to_idx")
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')

        # Add a third dim for Thermo (only has one channel in V2) so transformations can be applied
        if image_data.ndim == 2:
            image_data = image_data[:, :, np.newaxis]

        # change to accepted data type
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32)


        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label


class ASKIVIT_V2_metal_expert(Dataset):
    def __init__(self, directory: str, class_to_idx: dict, transform=None, modality=None):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .npy files.
        label_idx : dict
            Dictionary mapping label to index.
        transform : callable (optional)
            Transformation to apply to each image.
        """

        super(ASKIVIT_V2_metal_expert, self).__init__()
        self.directory = directory
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.modality = modality

        # List all .npy files in the directory / ASKIVIT dataset.
        self.filepaths_list = glob.glob(os.path.join(directory, f'{modality}_*.npy'))

        
    def __len__(self):
        return len(self.filepaths_list)
    
    def __getitem__(self, idx):
        # Get filepath to next .npy file.
        npy_filepath = self.filepaths_list[idx]

        """ Label extraction """
        # Get filename without path.
        npy_filename = os.path.basename(npy_filepath)
        
        # get label out of file name
        if "GT_" not in npy_filename:

            npy_filename_sliced = "_".join(npy_filename.split(".")[1].split("_")[3:])

            if "Non_wood_" in npy_filename_sliced:
                main_label = "_".join(npy_filename_sliced.split("_")[0:2])

                sub_label = "_".join(npy_filename_sliced.split("_")[2:])

            elif "Wood_" in npy_filename_sliced:
                main_label = "_".join(npy_filename_sliced.split("_")[0:1])

                sub_label = "_".join(npy_filename_sliced.split("_")[1:])

        # check subclass for metal and give mainclass based on that
        if "Metal" in sub_label:
            main_class_idx = 0
        elif "metal" in sub_label:
            main_class_idx = 0
        else:
            main_class_idx = 1
        
        """ Image loading """
        # Load numpy array directly from disk
        image_data = np.load(npy_filepath, mmap_mode='r')

        # Add a third dim for Thermo (only has one channel in V2) so transformations can be applied
        if image_data.ndim == 2:
            image_data = image_data[:, :, np.newaxis]

        # change to accepted data type
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32)

        """ Apply transformations to image if specified """
        if self.transform:
            label = torch.tensor(main_class_idx, dtype=torch.long)
            image_data = self.transform(image_data)

        return image_data, label

if __name__ == "__main__":
    directory = r"D:\data\Bild - ASKIVIT_V1\ASKVIT_V1_1\patches_fused_train_balanced_new"

    dataset = ASKIVIT_V1(directory)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=6, shuffle=True)

    print("Size of dataset: ", dataset.__len__())


    """
    Print collection of example images.
    """
    random.seed(42)
    num_samples = 9
    random_indices = random.sample(range(100), num_samples)

    cols, rows = 3, 3
    fig, ax = plt.subplots(cols, rows, figsize=(8, 8))
    ax = ax.flatten() # Easier assignment of axes by single index

    tick = time.time()
    samples_plotted = 0
    for i, (image, label, _) in enumerate(dataloader):
        if i in random_indices:
            img = image[0, :, :, :3].to(torch.uint8).numpy()
            img = img[..., [2, 1, 0]] # Convert from RGB to BGR

            assert img.shape == (50, 50, 3), f"Shape of image is {img.shape}"

            ax[samples_plotted].imshow(img)
            ax[samples_plotted].set_title(f"{label[0]}", fontsize=8)
            ax[samples_plotted].axis("off")

            samples_plotted += 1
            if samples_plotted == num_samples:
                break

    fig.tight_layout()

    tock = time.time()
    print(f"Time taken: {tock - tick:.2f} seconds")

    plt.show()