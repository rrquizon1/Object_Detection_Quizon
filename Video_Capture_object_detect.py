import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from PIL import Image
from torchvision import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import cv2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def get_model_instance_segmentation_load(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
num_classes=4
modelb = get_model_instance_segmentation_load(num_classes)
checkpoint = torch.load('Final_Model.pth')
modelb.load_state_dict(checkpoint['model_state_dict'])
# modelb.load_state_dict(torch.load('April222022b.pth'))
modelb.eval()

cap = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("Model_Demo.mp4", vid_cod, 15, (640,480))
#writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1280,960))
i = 0
modelb.to(device)
while(cap.isOpened()):
    ret, frame = cap.read()
     
    # This condition prevents from infinite looping
    # incase video ends.

    
  
    # Save Frame by Frame into disk using imwrite method
    # Define an initial bounding box
    # dim=(640,480)
    # frame=cv2.resize(frame,dim)
    frame_convert = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_convert = Image.fromarray(frame_convert)
    convert_tensor = transforms.ToTensor()
    frame_convert=convert_tensor(frame_convert)
    # time.sleep(1) 
    with torch.no_grad():
        prediction = modelb([frame_convert.to(device)])
    bbox=[]
    labels=[]
    for i in range(0,len(prediction[0]['boxes'])):
        xmin=[prediction[0]['boxes'][i][0]]
        ymin=[prediction[0]['boxes'][i][1]]
        xmax=[prediction[0]['boxes'][i][2]]
        ymax=[prediction[0]['boxes'][i][3]]
        labels=[prediction[0]['labels'][i]]
        # # bbox.append([xmin, ymin, xmax, ymax])
        # if prediction[0]['labels'][i]==1:
        #     cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),(255,0,0),2)
        #     cv2.putText(frame, 'Water', (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # if prediction[0]['labels'][i]==2:
        #     cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),(0,255,0),2)
        #     cv2.putText(frame, 'Soda', (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # if prediction[0]['labels'][i]==2:
        #     cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),(0,0,255),2)
        #     cv2.putText(frame, 'Juice', (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        if prediction[0]['labels'][i]==2:
            color=(0,0,255)
            text='Soda'
            cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),color,2)
            cv2.putText(frame, text, (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
        if prediction[0]['labels'][i]==1:
            color=(255,0,0)
            text='Water'
            cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),color,2)
            cv2.putText(frame, text, (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        if prediction[0]['labels'][i]==3:
            color=(0,255,0)
            text='Juice'
            cv2.rectangle(frame,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),color,2)
            cv2.putText(frame, text, (int(xmin[0]),int(ymin[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        
    # cv2.rectangle(frame,(100,100),(200,200),(0,255,0),2)
    # cv2.rectangle(frame, (650, 450), (420, 240),
    #           (255, 0, 0), 5)
    cv2.imshow('frame',frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
 
cap.release()
writer.release()
cv2.destroyAllWindows()