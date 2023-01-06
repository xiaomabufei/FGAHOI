import argparse

import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_path', type=str, default="/mnt/gluster/home/mashuailei/DAB-DETR-main/dahoi/param/pretrained_357_coco_obj.pth",
    )
    parser.add_argument(
        '--load_path', type=str, default="/mnt/gluster/home/mashuailei/DAB-DETR-main/dahoi/param/checkpoint0049_beforedrop.pth",
    )
    parser.add_argument(
        '--dataset', type=str, default='vcoco',
    )

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path, map_location='cpu')

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
               82, 84, 85, 86, 87, 88, 89, 90]

    # For no pair
    # obj_ids.append(91)
    for i in range(6):
        ps['model'][f'sub_bbox_embed.{i}.layers.0.weight'] = ps['model'][f'bbox_embed.{i}.layers.0.weight'].clone()
        ps['model'][f'sub_bbox_embed.{i}.layers.0.bias'] = ps['model'][f'bbox_embed.{i}.layers.0.bias'].clone()
        ps['model'][f'sub_bbox_embed.{i}.layers.1.weight'] = ps['model'][f'bbox_embed.{i}.layers.1.weight'].clone()
        ps['model'][f'sub_bbox_embed.{i}.layers.1.bias'] = ps['model'][f'bbox_embed.{i}.layers.1.bias'].clone()
        ps['model'][f'sub_bbox_embed.{i}.layers.2.weight'] = ps['model'][f'bbox_embed.{i}.layers.2.weight'].clone()
        ps['model'][f'sub_bbox_embed.{i}.layers.2.bias'] = ps['model'][f'bbox_embed.{i}.layers.2.bias'].clone()


        ps['model'][f'class_embed.{i}.weight'] = ps['model'][f'class_embed.{i}.weight'].clone()[obj_ids]
        ps['model'][f'class_embed.{i}.bias'] = ps['model'][f'class_embed.{i}.bias'].clone()[obj_ids]


    if args.dataset == 'vcoco':
        for i in range(6):
            l = nn.Linear(ps['model'][f'class_embed.{i}.weight'].shape[1], 2)
            l.to(ps['model'][f'class_embed.{i}.weight'].device)
            ps['model'][f'class_embed.{i}.weight'] = torch.cat((
                ps['model'][f'class_embed.{i}.weight'], l.weight))
            ps['model'][f'class_embed.{i}.bias'] = torch.cat(
                (ps['model'][f'class_embed.{i}.bias'], l.bias))

    torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)