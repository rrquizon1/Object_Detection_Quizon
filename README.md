# Object Detection using Fast R CNN by Rhodel Quizon

Beginner friendly repository for object detection. 
To be submitted as requirement 02 for EE298 Second Sem 2021-2022



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
pip install Image
pip install cv2
pip install os
pip install gdown
pip install tarfile
pip install sys
pip install Path
```

## Other requirements

Please take note that train.json and test.json are not the same with the given json files with the drinks dataset. Train.json and test.json are json files I made to conform to COCO dataset format. Check coco_maker2.py for the json file generator I made using SAHI library and the labels_train.csv and labels_test.csv.

Please make sure that there is no drinks folder or file within the working directory to download the model and the dataset. train.py and test.py checks the presence of drinks folder to download dataset. If it is present the dataset and the model file will not be downloaded.

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

To evaluate performance of the model on the test dataset of drinks dataset please run test.py. Test.py is tested to be running OK.

## Continue training a previously trained model

Run train.py. Check the variables below if there are problems during training. This .py file is tested to be running OK.

```

train_data_dir = r"drinks"
test_data_dir=r"drinks"
train_coco = r"train.json"
test_coco=r"test.json"

```

Other hyperparameters that can be changed are epochs, number of classes, and batch size. The values belwo are used to train the model

```
num_classes =4
num_epochs = 10
train_batch_size = 8

```

Learning rate scheduler below is used to train the final model:

```
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
```

You can modify lr, momentum, and weight decay parameters.

## Sample Result

For sample result, please download Model_demo.mp4. You may also use Video_Capture_object_detect.py to use your own webcam for object detection. 
