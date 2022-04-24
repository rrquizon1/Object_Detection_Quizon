# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:07:31 2022

@author: Lenovo
"""

import numpy as np

import csv
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image

with open('labels_train_a.csv', 'rt') as f:
    reader = csv.reader(f)
    data_as_list = list(reader)

# print(data_as_list)
coco = Coco()

coco.add_category(CocoCategory(id=1, name='water'))
coco.add_category(CocoCategory(id=2, name='soda'))
coco.add_category(CocoCategory(id=3, name='juice'))

for i in range(1,147):
    a=data_as_list[i]
    width, height = Image.open(a[0]).size
    coco_image = CocoImage(file_name=a[0], height=height, width=width,id=int(float(a[0][:-3])))
    coco_image.add_annotation(
    CocoAnnotation(
    bbox=[int(a[1]), int(a[3]), int(a[2]), int(a[4])],
    category_id=int(a[5]),
    category_name=a[6]
        )
    )

    coco.add_image(coco_image)


save_json(data=coco.json, save_path=r'D:\UPD MASters\EE 298\REQ02\coco make\annotations\test.json')

