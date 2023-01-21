# FGAHOI

FGAHOI: Fine-Grained Anchors for Human-Object Interaction Detection

## Abstract

Human-Object Interaction (HOI), as an important problem in computer vision, requires locating the human-object pair and identifying the interactive relationships between them. The HOI instance has a greater span in spatial, scale, and task than the individual object instance, making its detection more susceptible to noisy backgrounds. To alleviate the disturbance of noisy backgrounds on HOI detection, it is necessary to consider the input image information to generate fine-grained anchors which are then leveraged to guide the detection of HOI instances. However, it has the following challenges. 𝑖) how to extract pivotal features from the images with complex background information is still an open question. 𝑖𝑖) how to semantically align the extracted features and query embeddings is also a difficult issue. In this paper, a novel end-to-end transformer-based framework (FGAHOI) is proposed to alleviate the above problems. FGAHOI comprises three dedicated components namely, multi-scale sampling (MSS), hierarchical spatial-aware merging (HSAM) and task-aware merging mechanism (TAM). MSS extracts features of humans, objects and interaction areas from noisy backgrounds for HOI instances of various scales. HSAM and TAM semantically align and merge the extracted features and query embeddings in the hierarchical spatial and task perspectives in turn. In the meanwhile, a novel training strategy Stage-wise Training Strategy is designed to reduce the training pressure caused by overly complex tasks done by FGAHOI. In addition, we propose two ways to measure the difficulty of HOI detection and a novel dataset, 𝑖.𝑒., HOI-SDC for the two challenges (Uneven Distributed Area in Human-Object Pairs and Long Distance Visual Modeling of Human-Object Pairs) of HOI instances detection. Experiments are conducted on three benchmarks: HICO-DET, HOI-SDC and V-COCO. Our model outperforms the state-of-the-art HOI detection methods, and the extensive ablations reveal the merits of our proposed contribution.
![image](https://github.com/NEUfan/photo/blob/main/photo/727d4f2717cf7ca550470cb8954ebd0.jpg)

![image](https://github.com/NEUfan/photo/blob/main/photo/7c7447c6874b1a1bfe128310265a1f9.jpg)


![image](https://github.com/NEUfan/photo/blob/main/photo/ef02aebc48411c03633d9df29fb9168.jpg)

## Requirements

We test our models under ```python=3.8, pytorch=1.10.0, cuda=11.3```. Other versions might be available as well.

```bash
conda create -n FGAHOI python =3.8 pip
conda activate FGAHOI
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

- Compiling CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# test
python test.py
```

- The details of argument can be changed in ```main.py```
## Dataset Preparation&Result

### HICO-DET

Please follow the HICO-DET dataset preparation of [GGNet](https://github.com/SherlockHolmes221/GGNet).

After preparation, the `data/hico_20160224_det` folder as follows:

```bash
data
├── hico_20160224_det
|   ├── images
|   |   ├── test2015
|   |   └── train2015
|   └── annotations
|       ├── anno_list.json
|       ├── corre_hico.npy
|       ├── file_name_to_obj_cat.json
|       ├── hoi_id_to_num.json
|       ├── hoi_list_new.json
|       ├── test_hico.json
|       └── trainval_hico.json
```

### V-COCO

Please follow the installation of [V-COCO](https://github.com/s-gupta/v-coco).

For evaluation, please put `vcoco_test.ids` and `vcoco_test.json` into `data/v-coco/data` folder.

After preparation, the `data/v-coco` folder as follows:

```bash
data
├── v-coco
|   ├── prior.pickle
|   ├── images
|   |   ├── train2014
|   |   └── val2014
|   ├── data
|   |   ├── instances_vcoco_all_2014.json
|   |   ├── vcoco_test.ids
|   |   └── vcoco_test.json
|   └── annotations
|       ├── corre_vcoco.npy
|       ├── test_vcoco.json
|       └── trainval_vcoco.json
```
### HOI-SDC

After preparation, the `data/SDC` folder as follows:
```bash
data
├── SDC
|   ├── JPGImages
|   |   └── image
|   └── annotations
|       ├── train_annotation.json
|       ├── test_annotation.json
|       ├── train_split.txt
|       └── test_split.txt
```
More details will come soon!
## Results
We currently provide results on HICO-DET.

| Model | Full (def) | Rare (def) | None-Rare (def) | Full (ko) | Rare (ko) | None-Rare (ko) | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin-Tiny | 29.94 | 22.24 | 32.24 | 32.48 | 24.16 | 34.97 | [model] |
| Swin-Large*+ | 37.18 | 30.71 | 39.11 | 38.93 | 31.93 | 41.02 | [model] |

## Training

### HICO-DET
- Training FGAHOI with Swin-Tiny from scratch.

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --pretrained params/swin_tiny_patch4_window7_224.pth \
        --output_dir logs/swin_tiny_mul3 \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --use_nms
```

- Training FGAHOI with Swin-Large*+ from scratch.

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_large_384 \
        --pretrained params/swin_large_patch4_window12_384_22k.pth \
        --output_dir logs/swin_large_384_22k_mul3 \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --use_nms
```
### V-COCO
```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone [backbone_name] \
        --output_dir logs/[log_path] \
        --epochs 150 --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --use_nms [--no_obj]
```

- Train ResNet-50

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --backbone swin_tiny --pretrained params/swin_tiny_patch4_window7_224.pth --output_dir logs/swin_tiny_mul3_vcoco --epochs 150 --lr_drop 120 --num_feature_levels 3 --num_queries 300 --dataset_file vcoco --hoi_path data/v-coco --num_obj_classes 81 --num_verb_classes 29 --use_nms --no_obj
```
- Evaluation of V-COCO

Please generate the detection at first.

```bash
python generate_vcoco_official.py \
        --resume [checkpoint.pth] \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco \
        --dataset_file vcoco \
        --backbone [backbone_name] \
        --num_feature_level 3 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --use_nms [--no_obj]
```

Then, using the official code to evaluate.

```bash
python vsrl_eval.py --vcoco_path data/v-coco --detections vcoco.pickle
```
### HOI-SDC

- more details will come soon
## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
## Citation

If you use FGAHOI, please consider citing:
~~~
@inproceedings{Ma2023FGAHOI,
  title={FGAHOI: Fine-Grained Anchors forHuman-Object Interaction Detection},
  author={Shuailei Ma and Yuefeng Wang and Shanze Wang and Ying Wei},
  year={2023}
}
~~~
## Contact
Should you have any question, please contact {xiaomabufei@gmail.com} or {wangyuefeng0203@gmail.com}
**Acknowledgments:**

FGAHOI builds on previous works code base such as [QAHOI](https://github.com/cjw2021/QAHOI). If you found FGAHOI useful please consider citing these works as well.



