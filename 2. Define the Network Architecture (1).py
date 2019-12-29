# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

# import utilities to keep workspaces alive during model training
from workspace_utils import active_session

# watch for any changes in model.py, if it changes, re-load it automatically
%load_ext autoreload
%autoreload 2

## TODO: Define the Net in models.py
[IN][1]:
import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Once you've define the network, you can instantiate it
class Net(nn.Module):
    def __init__(self,weight):
        super(Net,self).__init__()
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(1,32,kernel_size = (k_height,k_width),bias = False)
        self.conv.weight = torch.nn.Parameter(weight)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        pooled_x = self.pool(activated_x)
        return conv_x,activated_x,pooled_x

        
# one example conv layer has been provided for you
from models import Net

net = Net()
print(net)
Net(
  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
)

[IN][2]:
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
crop = RandomCrop(224)
scale = Rescale(224)

Composed = transforms.Compose([Rescale(224),
                               RandomCrop(224)])

data_transform = Composed

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

[IN][3]:
    # create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())
    
Number of images:  3462
/home/workspace/data_load.py:39: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-19-aefbf393410b> in <module>()
      9 # iterate through the transformed dataset and print some stats about the first few samples
     10 for i in range(4):
---> 11     sample = transformed_dataset[i]
     12     print(i, sample['image'].size(), sample['keypoints'].size())
     13 

/home/workspace/data_load.py in __getitem__(self, idx)
     42 
     43         if self.transform:
---> 44             sample = self.transform(sample)
     45 
     46         return sample

/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/transforms.py in __call__(self, img)
     47     def __call__(self, img):
     48         for t in self.transforms:
---> 49             img = t(img)
     50         return img
     51 

/home/workspace/data_load.py in __call__(self, sample)
    132 
    133         top = np.random.randint(0, h - new_h)
--> 134         left = np.random.randint(0, w - new_w)
    135 
    136         image = image[top: top + new_h,

mtrand.pyx in mtrand.RandomState.randint (numpy/random/mtrand/mtrand.c:16117)()

ValueError: low >= high
