# Object Detection using Fast R CNN: Rhodel Quizon

Beginner friendly repository for object detection. 
To be submitted as requirement 02 for EE298 Second Sem 2021-2022



## References
This object detection is heavily based on the following tutorials:

[Pytorch Object detection Fine tuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

[How to train an object detector using COCO dataset](https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5)



## Fast R CNN

Fast R-CNN is an object detector neural network model developed by Ross Girshick.

It is a faster and more accurate implementation of the R-CNN. See image below for the fast R CNN Architecture



![ViTSTR Model](https://raw.githubusercontent.com/rrquizon1/Object_Detection_Quizon/main/figures/Fast%20R-CNN.png?token=GHSAT0AAAAAABTZPXKSSRCQNSS2ZFQV33YWYTEXZIA)

Paper:
[Arxiv](https://arxiv.org/abs/1504.08083)


## Install Requirements


```
pip install pycocotools
pip install torch
pip install torchvision
pip install PIL
pip install cv2
```


## Inference and Evaluation

To test the model please run Video_Capture_object_detect.py 
Modify this line with the location or file name of the model to be loaded

```
checkpoint = torch.load('Final_Model.pth')

```
