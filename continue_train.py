# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:06:39 2022

@author: Lenovo
"""
import sys
import os
library_path=os.getcwd() 

sys.path.append(library_path) #add additional modules


import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import vision.engine as engine
from PIL import Image
from torchvision import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import transforms as T
import utils
from engine import train_one_epoch, evaluate


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


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

import utils
train_data_dir = r"Images"
test_data_dir=r"Images"
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
train_batch_size = 1

# own DataLoader
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(my_dataset_test,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model_instance_segmentation_load(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch
  
num_classes =4
num_epochs = 10
model = get_model_instance_segmentation(num_classes)
model.to(device)   
# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

#len_dataloader = len(data_loader)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


model,optimizer,epoch=load_checkpoint(model,optimizer,'April252022.pth')

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    if __name__ == '__main__': 
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
        lr_scheduler.step()
    # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


save_checkpoint(model,optimizer,"Model.pth",epoch)
