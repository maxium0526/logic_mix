from torchvision import transforms
import numpy as np
import os
import pandas as pd
import xmltodict
import json
import glob
from .core import *

def MSCOCO(dataset_path, year='2014', split='train', transform=transforms.ToTensor()):
    from pycocotools.coco import COCO

    num_categories = 80

    if split == 'train':
        subset = 'train'
    if split == 'valid':
        subset = 'val'

    coco = COCO(os.path.join(dataset_path, 'annotations', f'instances_{subset}{year}.json'))
    all_category_ids = coco.getCatIds()

    records = []
    image_ids = coco.getImgIds()
    for i, img_id in enumerate(image_ids):
        print(f'Loading MSCOCO {split}: {i+1} / {len(image_ids)}', end='\r')
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(subset+year, img_filename)
        pos_category_ids = [coco.loadAnns(annotation_id)[0]['category_id'] for annotation_id in coco.getAnnIds(imgIds=img_id)]
        pos_category_ids = list(set(pos_category_ids))
        pos_category_nos = [all_category_ids.index(category_id) for category_id in pos_category_ids]
        pos_category_nos.sort()
        records.append((img_id, path, pos_category_nos, [], []))
    print()
    
    records = fill_nan_to_negative(records, num_categories)

    return MLCPLDataset(dataset_path, records, num_categories, transform)

def Pascal_VOC_2007(dataset_path, split='train', transform=transforms.ToTensor()):

    if split == 'train':
        subset = 'trainval'
    elif split == 'valid':
        subset = 'test'

    all_category_ids = set({})
    paths = glob.glob(os.path.join(dataset_path, 'ImageSets', 'Main', '*.txt'))
    for i, path in enumerate(paths):
        print(f'Finding categories of Pascal VOC 2007: {i+1} / {len(paths)}', end='\r')
        basename = os.path.basename(path)
        if '_' in basename:
            all_category_ids.add(basename.split('_')[0])
    all_category_ids = list(all_category_ids)
    all_category_ids.sort()
    num_categories = len(all_category_ids)

    img_nos = pd.read_csv(os.path.join(dataset_path, 'ImageSets', 'Main', subset+'.txt'), sep=' ', header=None, names=['Id'], dtype=str)
    records = []
    for i, row in img_nos.iterrows():
        print(f'Loading Pascal VOC 2007 {split}: {i+1} / {img_nos.shape[0]}', end='\r')
        img_no = row['Id']
        path = os.path.join('JPEGImages', img_no+'.jpg')

        xml_path = os.path.join(dataset_path, 'Annotations', f'{img_no}.xml')
        with open(xml_path, 'r') as f:
            data = f.read()
        xml = xmltodict.parse(data)
        detections = xml['annotation']['object']
        if isinstance(detections, list):
            pos_category_ids = list(set([detection['name'] for detection in detections]))
            pos_category_ids.sort()
        else:
            pos_category_ids = [detections['name']]
        
        pos_category_nos = [all_category_ids.index(i) for i in pos_category_ids]
        records.append((img_no, path, pos_category_nos, [], []))

    records = fill_nan_to_negative(records, num_categories)

    return MLCPLDataset(dataset_path, records, num_categories, transform)

def VG_200(dataset_path, metadata_path=None, split='train', transform=transforms.ToTensor()):

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'test'

    metadata_path = dataset_path if metadata_path is None else metadata_path

    num_categories = 200

    vg_folder_1 = 'VG_100K'
    vg_folder_2 = 'VG_100K_2'

    folder_1 = os.listdir(os.path.join(dataset_path, vg_folder_1))
    folder_2 = os.listdir(os.path.join(dataset_path, vg_folder_2))

    records = []

    image_ids = pd.read_csv(os.path.join(metadata_path, f'{subset}_list_500.txt'), header=None)[0].tolist()

    with open(os.path.join(metadata_path, 'vg_category_200_labels_index.json'), 'r') as f:
        labels = json.load(f)

    for i, image_id in enumerate(image_ids):
        print(f'Loading VG_200 ({split}): {i+1} / {len(image_ids)}', end='\r')

        positives = labels[image_id]

        if image_id in folder_1:
            folder = vg_folder_1
        elif image_id in folder_2:
            folder = vg_folder_2
        else:
            raise Exception(f'Image {image_id} not found.')

        img_path = os.path.join(dataset_path, folder, f'{image_id}')
        records.append((image_id, img_path, positives, [], []))

    print()

    records = fill_nan_to_negative(records, num_categories=num_categories)

    return MLCPLDataset(dataset_path, records, num_categories, transform=transform)