"""
HAKE detection dataset.
"""
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import torchvision
import math
import torch.nn.functional as F

import datasets.transforms_hake as T
import os

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class HAKEDetection(torch.utils.data.Dataset):

    def __init__(self, root, image_split, img_set, img_folder, anno_file, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        self.image_split = image_split
        self.file_names = self.extract_fns(image_split, root)
        with open(root / 'proposed_challenge' / 'dc_hoi_list.json', 'r') as f:
            self.hoi_anno = json.load(f)
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.num_queries = num_queries
        self._valid_verb_ids = list(range(0, 93))
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.file_names[idx]]

        img = Image.open(self.img_folder / self.file_names[idx]).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['labels']) > self.num_queries:
            img_anno['labels'] = img_anno['labels'][:self.num_queries]

        human_boxes = [obj['human_bbox'] for obj in img_anno['labels']]
        # guard against no boxes via resizing
        human_boxes = torch.as_tensor(human_boxes, dtype=torch.float32).reshape(-1, 4) #x y x y
        object_boxes = [obj['object_bbox'] for obj in img_anno['labels']]
        # guard against no boxes via resizing
        object_boxes = torch.as_tensor(object_boxes, dtype=torch.float32).reshape(-1, 4)#x y x y
        human_label = [0 for obj in img_anno['labels']]
        obj_label = [self.hoi_anno[obj['hoi_id']-1]['object_index'] for obj in img_anno['labels']] #bu yong jian yi yin wei zhu shi zhong jiu shi cong ling kai shi
        obj_labels = torch.tensor(obj_label, dtype=torch.int64)
        verb = [self.hoi_anno[obj['hoi_id']-1]['verb_id'] for obj in img_anno['labels']] #xu yao jian yi yin wei zhu shi cong yi kai shi
        verb_labels = torch.tensor(verb, dtype=torch.int64)
        verb_labels = F.one_hot(verb_labels, num_classes=len(self._valid_verb_ids))
        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            # qu chu biao zhu cuo wu de kuang
            human_boxes[:, 0::2].clamp_(min=0, max=w)
            human_boxes[:, 1::2].clamp_(min=0, max=h)
            object_boxes[:, 0::2].clamp_(min=0, max=w)
            object_boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (human_boxes[:, 3] > human_boxes[:, 1]) & (human_boxes[:, 2] > human_boxes[:, 0])
            human_boxes = human_boxes[keep]
            object_boxes = object_boxes[keep]
            obj_labels = obj_labels[keep]
            verb_labels = verb_labels[keep]
            keep = (object_boxes[:, 3] > object_boxes[:, 1]) & (object_boxes[:, 2] > object_boxes[:, 0])
            human_boxes = human_boxes[keep]
            object_boxes = object_boxes[keep]
            obj_labels = obj_labels[keep]
            verb_labels = verb_labels[keep]
            boxes = torch.cat((human_boxes,object_boxes), dim=0)
            target['boxes'] = boxes
            target['keep'] = torch.tensor([True for i in range(len(boxes))], device=boxes.device)
            target['labels'] = torch.cat((torch.zeros_like(obj_labels), obj_labels), dim=0)
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if self._transforms is not None:
                img, target = self._transforms(img, target) # x y x y -> x y w h
            index = len(target['boxes'])//2
            if len(obj_labels) == 0:
                # target['name'] = self.file_names[idx]
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['verb_label_enc'] = torch.zeros(len(self._valid_verb_ids), dtype=torch.float32)
            else:
                # target['name'] = self.file_names[idx]
                target['obj_labels'] = target['labels'][index:].to(torch.int64)
                verb_keep = target['keep'][:len(boxes)//2]
                target['verb_labels'] = verb_labels[verb_keep].to(torch.float32)
                target.pop('keep')
                target['sub_boxes'] = target['boxes'][:index].to(torch.float32)
                target['obj_boxes'] = target['boxes'][index:].to(torch.float32)
                target['verb_label_enc'] = verb_labels[-1].to(torch.float32)
                assert target['obj_boxes'].shape[0] == target['sub_boxes'].shape[0], f'loading dataset occupies error'
        else:
            target['boxes'] = torch.cat((human_boxes,object_boxes), dim=0)
            target['labels'] = torch.cat((torch.zeros_like(obj_labels), obj_labels), dim=0)
            target['id'] = idx
            target['file_name'] = self.file_names[idx]

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = list(zip(human_label, obj_label, verb))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)
    
    def extract_fns(self, image_split, voc_root):
        splits_dir = os.path.join(voc_root, 'proposed_challenge')
        split_f = os.path.join(splits_dir, image_split.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

# Add color jitter to coco transforms
def make_hake_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hake_path)
    if image_set == 'train':
        image_split = args.hake_split_train
    else:
        image_split = args.hake_split_test
    assert root.exists(), f'provided HOI path {root} does not exist'
    if image_set == 'train':
        PATHS = {
            'path': (root / 'JPEGImages' , root / 'proposed_challenge' / 'train_annotation.json')
        }
    else:
        PATHS = {
            'path': (root / 'JPEGImages' , root / 'proposed_challenge' / 'test_annotation.json')
        }
    CORRECT_MAT_PATH = root / 'proposed_challenge' / 'corre_dc.npy'
    img_folder, anno_file = PATHS['path']
    dataset = HAKEDetection(root, image_split, image_set, img_folder, anno_file, transforms=make_hake_transforms(image_set),
                            num_queries=args.num_queries)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
