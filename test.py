# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:41:39 2022

@author: Lenovo
"""




import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import sys
library_path=os.getcwd() 
import gdown
import tarfile
from pathlib import Path

def is_valid_directory(filename):
    p = Path(filename)
    return p.exists() and p.is_dir()

fp=is_valid_directory('drinks')

if fp is False:
    url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
    output = 'drinks.tgz'
    gdown.download(url, output, quiet=False)

    url = 'https://drive.google.com/uc?id=1uRyrOCXAHaSb-po2ESRCQ4Zvv4Nm-dPY'
    output = 'Final_Model.pth'
    gdown.download(url, output, quiet=False)

# open file
    file = tarfile.open('drinks.tgz')
  
# extracting file
    file.extractall('')
  
    file.close()

sys.path.append(library_path) #add additional modules
class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels=[]
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            label= coco_annotation[i]['category_id']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
           
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)
        # Labels (In my case, I only one class: target class or background)
        #labels=coco.getCatIds(imgIds=img_id)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


train_data_dir = r"drinks"
test_data_dir=r"drinks"
train_coco = r"train.json"
test_coco=r"test.json"
def collate_fn(batch):
    return tuple(zip(*batch))
# create own Dataset
my_dataset = myOwnDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )
my_dataset_test = myOwnDataset(root=test_data_dir,
                          annotation=test_coco,
                          transforms=get_transform()
                          )



# Batch size
train_batch_size = 8

# own DataLoader
data_loader_test = torch.utils.data.DataLoader(my_dataset_test,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation_load(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

num_classes =4
num_epochs = 10
model = get_model_instance_segmentation_load(num_classes)
checkpoint = torch.load('Final_Model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# modelb.load_state_dict(torch.load('April222022b.pth'))
model.eval()
model.to(device)      
if __name__ == '__main__': 
    evaluate(model, data_loader_test, device=device)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')