# FGAHOI
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fgahoi-fine-grained-anchors-for-human-object/human-object-interaction-detection-on-hico)](https://paperswithcode.com/sota/human-object-interaction-detection-on-hico?p=fgahoi-fine-grained-anchors-for-human-object)
[![arXiv](https://img.shields.io/badge/arXiv-2301.04019-b31b1b.svg)](https://arxiv.org/abs/2301.04019)

FGAHOI: Fine-Grained Anchors for Human-Object Interaction Detection

## Abstract

Human-Object Interaction (HOI), as an important problem in computer vision, requires locating the human-object pair and identifying the interactive relationships between them. The HOI instance has a greater span in spatial, scale, and task than the individual object instance, making its detection more susceptible to noisy backgrounds. To alleviate the disturbance of noisy backgrounds on HOI detection, it is necessary to consider the input image information to generate fine-grained anchors which are then leveraged to guide the detection of HOI instances. However, it has the following challenges. 𝑖) how to extract pivotal features from the images with complex background information is still an open question. 𝑖𝑖) how to semantically align the extracted features and query embeddings is also a difficult issue. In this paper, a novel end-to-end transformer-based framework (FGAHOI) is proposed to alleviate the above problems. FGAHOI comprises three dedicated components namely, multi-scale sampling (MSS), hierarchical spatial-aware merging (HSAM) and task-aware merging mechanism (TAM). MSS extracts features of humans, objects and interaction areas from noisy backgrounds for HOI instances of various scales. HSAM and TAM semantically align and merge the extracted features and query embeddings in the hierarchical spatial and task perspectives in turn. In the meanwhile, a novel training strategy Stage-wise Training Strategy is designed to reduce the training pressure caused by overly complex tasks done by FGAHOI. In addition, we propose two ways to measure the difficulty of HOI detection and a novel dataset, 𝑖.𝑒., HOI-SDC for the two challenges (Uneven Distributed Area in Human-Object Pairs and Long Distance Visual Modeling of Human-Object Pairs) of HOI instances detection. Experiments are conducted on three benchmarks: HICO-DET, HOI-SDC and V-COCO. Our model outperforms the state-of-the-art HOI detection methods, and the extensive ablations reveal the merits of our proposed contribution.


![图片1](https://user-images.githubusercontent.com/104605826/215409373-63b4ade9-2ff2-41cb-b65d-08c7f3d78954.png)


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
cd ./models/dab_deformable_detr/ops
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

| Model | Full (def) | Rare (def) | None-Rare (def) | Full (ko) | Rare (ko) | None-Rare (ko) |Weight|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin-Tiny | 29.94 | 22.24 | 32.24 | 32.48 | 24.16 | 34.97 |[Tiny_weight](https://drive.google.com/file/d/1WzBZ-jEy7kA02ynXQdugnF44e7cTNnD9/view?usp=sharing)|
| Swin-Large*+ | 37.18 | 30.71 | 39.11 | 38.93 | 31.93 | 41.02 |[Large_weight](https://drive.google.com/file/d/1GTm9vh1D145wJhlZXpDGJMMUnWi-_gM_/view?usp=sharing)|

## Training
### HICO-DET
- Training FGAHOI with Swin-tiny from scratch.

stage base
```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --pretrained params/swin_tiny_patch4_window7_224.pth \
        --dataset_file hico \
        --num_verb_classes 117 \
        --num_obj_classes 80 \
        --output_dir logs/base \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --merge \
        --scale [1, 3, 5] \
        --base \
        --hoi_path data/hico_20160224_det
```

stage hierarchical_merge
```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --pretrain_model_path "{Weights of the last stage}" \
        --dataset_file hico \
        --num_verb_classes 117 \
        --num_obj_classes 80 \
        --output_dir logs/hierarchical_merge \
        --epochs 50 \
        --lr_drop 40 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --merge \
        --scale [1, 3, 5] \
        --hierarchical_merge \
        --hoi_path data/hico_20160224_det

```

stage hierarchical_merge and task_merge
```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --pretrain_model_path "{Weights of the last stage}" \
        --dataset_file hico \
        --num_verb_classes 117 \
        --num_obj_classes 80 \
        --output_dir logs/hierarchical_merge_and_task_merge \
        --epochs 50 \
        --lr_drop 40 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --merge \
        --scale [1, 3, 5] \
        --hierarchical_merge \  
        --task_merge \
        --hoi_path data/hico_20160224_det
```
## Testing
- Evaluate FGAHOI with Swin-tiny from scratch.
```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --dataset_file hico \
        --resume "{Weight of the model}"
        --num_verb_classes 117 \
        --num_obj_classes 80 \
        --output_dir logs \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --merge \
        --scale [1, 3, 5] \
        --hierarchical_merge \  
        --task_merge \
        --eval \
        --hoi_path data/hico_20160224_det
```

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
Should you have any question, please contact {xiaomabufei@gmail.com}

## Acknowledgments

FGAHOI builds on previous works code base such as [QAHOI](https://github.com/cjw2021/QAHOI), [DAB-DETR](https://github.com/xiaomabufei/DAB-DETR). If you found FGAHOI useful please consider citing these works as well.



