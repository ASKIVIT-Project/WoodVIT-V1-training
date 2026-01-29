import numpy as np
import torch
from torchvision.transforms import v2


class SelectChannels(object):
    """
    Select subset of channels from image to be used.
    # object: list(P["SELECTED_CH"])
    """
    def __init__(self, selected_channels: list):
        self.selected_channels = selected_channels
        
    def __call__(self, image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        return image[:, :, self.selected_channels]
    
    def __repr__(self):
        first_ch = self.selected_channels[0]
        last_ch = self.selected_channels[-1]
        print_channels = f"channels=[{first_ch}-{last_ch}]"
        return f"{self.__class__.__name__}({print_channels})"


class AdjustChannelRange(torch.nn.Module):
    # NOT USED ATM
    # usage not clear
    def __init__(self, max_value, thz_start_channel):
        super(AdjustChannelRange, self).__init__()
        self.max_value = max_value
        self.start_channel = thz_start_channel

    def forward(self, x):
        # Apply absolute value only to the selected channels
        x[self.start_channel:, :, :] = torch.abs(x[self.start_channel:, :, :])
        
        # Adjust the selected channels by dividing by the maximum value
        x[self.start_channel:, :, :] = x[self.start_channel:, :, :] / self.max_value
        
        # Clamp values to ensure they are within [0, 1] range
        x = torch.clamp(x, min=0, max=1)
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BGR2RGB(torch.nn.Module):
    def __init__(self):
        super(BGR2RGB, self).__init__()
    
    def forward(self, x):
        return x[:, :, [2, 1, 0]]
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class NormalizeWrapped(object):
    """
    Only purpose is to wrap torchvision.transforms.Normalize to be able
    to overwrite __repr__ method, to NOT print all mean and std values.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std

        self.normalize = v2.Normalize(mean, std, inplace)
    
    def __call__(self, img):
        return self.normalize(img)
    
    def __repr__(self):
        if len(self.mean) < 4:
            return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
        else:
            return f"{self.__class__.__name__}(mean='[...]', std='[...]')"
    

class ToTensor(object):
    """
    Convert np.ndarrays in sample to Tensors.
    
    """
    def __call__(self, image):
        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.tensor(image).float()
    


class NormalizeNumpy(object):
    """
    Normalize the input numpy array image with the given min and max values to 
    the range [0, 1].
    """
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

        if self.min_val >= self.max_val:
            raise ValueError(f"min_val={self.min_val} must be less than max_val={self.max_val}")

    def __call__(self, image):     
        normalized_image = (image - self.min_val) / (self.max_val - self.min_val)
        return normalized_image

    def __repr__(self):
        return f"{self.__class__.__name__}(min_val={self.min_val}, max_val={self.max_val})"
