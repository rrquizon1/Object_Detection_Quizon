# Object Detection using Fast R CNN by Rhodel Quizon

Beginner friendly repository for object detection. 
To be submitted as requirement 02 for EE298 Second Sem 2021-2022



## References
This object detection is heavily based on the following tutorials:

[Pytorch Object detection Fine tuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

[How to train an object detector using COCO dataset](https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5)



## Fast R CNN

Fast R-CNN is an object detector neural network model developed by Ross Girshick.

It is a faster and more accurate implementation of the R-CNN. See image below for the fast R CNN Architecture



![FAST R CNN Model](https://i.ibb.co/tX22fB9/FASTRCNN.png)

Paper:
[Arxiv](https://arxiv.org/abs/1504.08083)


## Install Requirements


```
pip install pycocotools
pip install torch
pip install torchvision
pip install PIL
pip install cv2
pip install os
```

## Other requirements

Please take note that train.json and test.json is not the same with the given json files with the drinks dataset. Train.json and test.json are json files I made to conform to COCO dataset format. Check coco_maker2.py for the json file generator I made using SAHI library.

## Inference and Evaluation

To test the model please run Video_Capture_object_detect.py 
Modify this line with the location and file name of the model to be loaded

```
checkpoint = torch.load('Final_Model.pth')

```

This will run a program that captures video from webcam and detect the three objects required for detection.

You can also test the model using images. Use modeleval.py and change the same variable edited above for model location.
Change the variable below with the location of image to be evaluated.

```
imgpath='WIN_20220422_09_04_12_Pro.jpg'

```

## Training Fast R-CNN from scratch Model

Run train_start.py but change the following variables



```

train_data_dir = r"D:\UPD MASters\EE 298\REQ02\Final Files\Images" # Training Images directory
test_data_dir=r"D:\UPD MASters\EE 298\REQ02\Final Files\Images" # Testing Images directory
train_coco = r"train.json" # Train annotations
test_coco=r"test.json" #Test annotation
```

Other hyperparameters that can be changed are epochs, number of classes, and batch size

```
num_classes =4 #number of classes put here are 4 1 for water,2 for soda,3 for juice, and 0 for background therefore 4 classes.
num_epochs = 10
train_batch_size = 8

```





## Continue training a previously trained model

Run continue_train.py but change the following variables

Run all blocks. Dont forget to change the directory for train and test dataset and annotations.

```

train_data_dir = r"D:\UPD MASters\EE 298\REQ02\Final Files\Images" # Training Images directory
test_data_dir=r"D:\UPD MASters\EE 298\REQ02\Final Files\Images" # Testing Images directory
train_coco = r"train.json" # Train annotations
test_coco=r"test.json" #Test annotation

```

Other hyperparameters that can be changed are epochs, number of classes, and batch size

```
num_classes =4
num_epochs = 10
train_batch_size = 8

```

