# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import torch
import torch.nn.functional as F
from torch import nn
import math
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
import einops
import numpy as np
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def dis(x):
    return x[-1]
def sample(num, points):
    if num == 0:
        result = []
    else:
        while len(points) < num:       
            points.append(([0, 0], 0))
        space = int(len(points)/num)
        points.sort(key=dis)
        m = 0
        sample_points = []
        for i in range(num):
            sample_points.append(np.array(points[m+i*space][0]))
        result = np.array(sample_points)
    return result

def Radar_sampling(num_query, N, points_all):
    all_sample_points = []
    for points in points_all:
        points = [x.tolist() for x in points]
        N_points = len(points)
        x = 0
        y = 0
        for point in points:
            x = x + point[0] / len(points)
            y = y + point[1] / len(points)
        list_distance = []
        for i in range(len(points)):
            list_distance.append(math.sqrt((points[i][0]-x)**2+(points[i][1]-y)**2))
        L = max(list_distance)
        new_points = list(zip(points, list_distance))
        num = []
        D = []
        for i in range(1, N+1):
            D.append([point for point in new_points if point[-1] < L*i/N and point[-1] >= L*(i-1)/N])
            num.append(len(D[-1]))
        new_num = [int(n*num_query/N_points) for n in num[:-1]]
        new_num.append(num_query - sum(new_num))
        sample_points = []
        for i in range(N):
            sample_points.extend(sample(new_num[i], D[i]))
        all_sample_points.append(sample_points)
    return all_sample_points




class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
class DABDeformableDETR(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, args, backbone, transformer, num_classes, num_verb_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 use_dab=False, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 no_obj=False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes) if not no_obj else nn.Linear(hidden_dim, num_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.hoi_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.hoi_score = nn.Linear(hidden_dim, 1)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
                

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.dynamic = args.dynamic
        self.ratios = args.ratios
        self.scales = args.scales
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if not no_obj:
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        else:
            self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        # self.hoi_score.bias.data = torch.ones(1)*bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.hoi_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.hoi_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.hoi_bbox_embed.layers[-1].bias.data[2:], -2.0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            # self.hoi_score = _get_clones(self.hoi_score, num_pred)
            # zan ding zhi you hoi de kuang quan dou wei tiao
            self.hoi_bbox_embed = _get_clones(self.hoi_bbox_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            nn.init.constant_(self.hoi_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.class_embed = None
            self.transformer.decoder.verb_embed = None
            self.transformer.decoder.bbox_embed = None
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.hoi_bbox_embed = self.hoi_bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.hoi_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            # self.hoi_score = nn.ModuleList([self.hoi_score for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.hoi_bbox_embed = nn.ModuleList([self.hoi_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.class_embed = None
            self.transformer.decoder.verb_class_embed = None
            self.transformer.decoder.bbox_embed = None
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.hoi_bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        # get dynamic anchor
        self.query_lvl = self.num_queries // 3 # the query num of each level feature
        # get reference points offset
        offset = []
        num_channels = backbone.num_channels
        self.offset_stride = args.offset_stride
        for lvl in range(len(backbone.num_channels)):
            channels = num_channels[lvl]
            strides = self.offset_stride[lvl]
            offset.append(nn.Sequential(
                    nn.Conv2d(channels, channels, strides, strides, groups=channels),
                    LayerNormProxy(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, 2, 1, 1, 0, bias=False)
                ))
        self.offset = nn.ModuleList(offset)
    def anchor_to_boxes(self, anchor, ratios=[0.5,1,2],scales=[0.1,0.2,0.4]):
        #input B N 2
        #output B N 9 4
        size = []
        for r in ratios:
            for s in scales:
                size.append([s*r,s/r])
        size = torch.tensor(size, dtype=torch.float32, device=anchor.device).unsqueeze(0).unsqueeze(0).repeat(anchor.shape[0], anchor.shape[1],1,1) #B*N*9*2
        boxes = anchor.unsqueeze(2).repeat(1,1,9,2) # B N 9 4
        boxes[:,:,:,2:] = size[:,:,:,:] # x y w h
        B, N, k, _ = boxes.shape
        boxes = einops.rearrange(boxes, 'B N K C -> (B N K) C')
        boxes = box_cxcywh_to_xyxy(boxes) # x y x y
        boxes[boxes<0]=0
        boxes[boxes>1]=1
        boxes = einops.rearrange(boxes, '(B N K) C ->B N K C',B=B,N=N,K=k)
        return box_xyxy_to_cxcywh(boxes) # x y w h 
    def generate_boxes(self, anchor, ratios=[0.5,1,2],scales=[0.1,0.2,0.4]):
        boxes = {}
        for key, value in anchor.items():
            boxes[key] = self.anchor_to_boxes(value)
        return boxes
    def get_anchor(self, features):
        name = ['low', 'middle', 'high']
        dynamic_anchors = {}
        for l, feat in enumerate(features):
            tensor, mask = feat.decompose()
            shift = self.offset[l](tensor) # B * 2 * H/S * W/S
            # H, W = tensor.shape[-2], tensor.shape[-1]
            coords_h = torch.linspace(0, 1, tensor.shape[-2]//self.offset_stride[l], dtype=torch.float32, device=shift.device)
            coords_w = torch.linspace(0, 1, tensor.shape[-1]//self.offset_stride[l], dtype=torch.float32, device=shift.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            reference = coords.unsqueeze(0).repeat(shift.shape[0],1,1,1) # B 2 H W
            anchor = reference + shift
            anchor = einops.rearrange(anchor, 'B C H W -> B (H W) C') # B * (H/S * W/S) * 2
            dynamic_anchors[name[l]] = anchor
        # dynamic_anchor = Radar_sampling(self.num_queries, 5, dynamic_anchor) # B * NUM_QUERIES * 2
        return dynamic_anchors
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        if self.dynamic:
            #   Plan 1
            dynamic_anchor_boxes = self.generate_boxes(self.get_anchor(features), ratios=self.ratios, scales=self.scales) # x y x y
        else:
            dynamic_anchor_boxes = None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # import ipdb; ipdb.set_trace()

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 2
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                # multi patterns
                tgt_embed = self.tgt_embed.weight           # nq, 256
                pat_embed = self.patterns_embed.weight      # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references_out, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds,dynamic_anchor_boxes)


        outputs_classes = []
        outputs_verb_classes = []
        outputs_coords = []
        outputs_sub_coords = []
        outputs_hoi_coords = []
        outputs_references = []
        # outputs_hoi_scores = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references_out[lvl-1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_verb_class = self.verb_class_embed[lvl](hs[lvl])
            # output_hoi_score = self.hoi_score[lvl](hs1[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp_sub = self.sub_bbox_embed[lvl](hs[lvl])
            tmp_hoi = self.hoi_bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
                tmp_sub += reference
                tmp_hoi += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp_sub[..., :2] += reference
                tmp_hoi[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_sub_coord = tmp_sub.sigmoid()
            outputs_hoi_coord = tmp_hoi.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_verb_classes.append(outputs_verb_class)
            # outputs_hoi_scores.append(output_hoi_score)
            outputs_coords.append(outputs_coord)
            outputs_sub_coords.append(outputs_sub_coord)
            outputs_hoi_coords.append(outputs_hoi_coord)
            reference = reference.sigmoid()
        outputs_class = torch.stack(outputs_classes)
        # outputs_hoi_score = torch.stack(outputs_hoi_scores)
        outputs_coord = torch.stack(outputs_coords)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_hoi_coord = torch.stack(outputs_hoi_coords)
        outputs_verb_class = torch.stack(outputs_verb_classes)

        out = {'pred_obj_logits': outputs_class[-1], 'pred_obj_boxes': outputs_coord[-1],
                'pred_verb_logits': outputs_verb_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1], 'pred_hoi_boxes': outputs_hoi_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_verb_class, outputs_sub_coord, outputs_hoi_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_verb_class, outputs_sub_coord, outputs_hoi_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_obj_boxes': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_hoi_boxes': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], outputs_verb_class[:-1], outputs_sub_coord[:-1], outputs_hoi_coord[:-1])]


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25,
                 no_obj=False):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.no_obj = no_obj
        self.distance = torch.nn.PairwiseDistance(p=2)
        if self.no_obj:
            empty_weight = torch.ones(self.num_obj_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)
    def loss_hoi_score_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_hoi_score' in outputs
        pred_hoi_score = outputs['pred_hoi_score']   
        idx = self._get_src_permutation_idx(indices)
        target_hoi_score = torch.zeros_like(pred_hoi_score)
        target_hoi_score[idx] = torch.tensor([1], dtype=torch.float32, device=pred_hoi_score.device)
        loss_hoi_score_ce = sigmoid_focal_loss(pred_hoi_score, target_hoi_score, num_interactions, alpha=self.focal_alpha, gamma=2) * pred_hoi_score.shape[1]
        losses = {'loss_hoi_score_ce': loss_hoi_score_ce}
        return losses
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if not self.no_obj:
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_obj_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_interactions, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        
        loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses
    def loss_anchor(self, outputs, targets, indices, num_interactions):
        assert 'dynamic_anchor' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_dynamic_anchor = outputs['dynamic_anchor'][idx]
        target_sub_points = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_points = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_interaction_points = (target_obj_points[:,:2] + target_sub_points[:,:2]) / 2
        distance = self.distance(src_dynamic_anchor, target_interaction_points)
        loss = distance.mean()
        losses = {'loss_anchor': loss}
        return losses
    def hoi_box(self, sub_boxes, obj_boxes):
        # sub_boxes  x y w h
        # obj_boxes  x y w h
        #return hoi_boxes x y w h
        h = box_cxcywh_to_xyxy(sub_boxes)
        o = box_cxcywh_to_xyxy(obj_boxes)
        hoi_boxes = torch.tensor([[min(h[i][0],o[i][0]),min(h[i][1],o[i][1]),max(h[i][2],o[i][2]),max(h[i][3],o[i][3])] for i in range(sub_boxes.shape[0])], 
                                    dtype=torch.float32, device=sub_boxes.device)    # x y x y
        if hoi_boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32, device=sub_boxes.device)
        return box_xyxy_to_cxcywh(hoi_boxes) # x y w h
    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]  # x y w h
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        # src_hoi_boxes = outputs['pred_hoi_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_hoi_boxes = self.hoi_box(target_sub_boxes, target_obj_boxes)
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)
        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            # losses['loss_hoi_bbox'] = src_hoi_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
            # losses['loss_hoi_giou'] = src_hoi_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            # loss_hoi_bbox = F.l1_loss(src_hoi_boxes, target_hoi_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            # losses['loss_hoi_bbox'] = loss_hoi_bbox.sum() / num_interactions
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            # loss_hoi_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_hoi_boxes),
            #                                                    box_cxcywh_to_xyxy(target_hoi_boxes)))                                                   
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
            # losses['loss_hoi_giou'] = loss_hoi_giou.sum() / num_interactions
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if not loss == "verb_label_enc":
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "verb_label_enc":
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            if "verb_label_enc" in self.losses:
                l_dict = self.get_loss("verb_label_enc", enc_outputs, targets, None, None)
                l_dict = {k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id, no_obj=False):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.no_obj = no_obj

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']



        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        verb_scores = out_verb_logits.sigmoid()
        num_verb_classes = verb_scores.shape[-1]
        
        # top 100
        obj_prob_class_all = obj_prob[:, :, :-1] if self.no_obj else obj_prob
        num_obj_classes = obj_prob_class_all.shape[-1]

        topk_values, topk_indexes = torch.topk(obj_prob_class_all.flatten(1), 100, dim=1)
        obj_scores = topk_values
        topk_boxes = topk_indexes // num_obj_classes
        obj_labels = topk_indexes % num_obj_classes

        # top 100
        verb_scores = torch.gather(verb_scores, 1, topk_boxes.unsqueeze(-1).repeat(1,1,num_verb_classes))
        out_obj_boxes = torch.gather(out_obj_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        out_sub_boxes = torch.gather(out_sub_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
        
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_dab_deformable_detr(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DABDeformableDETR(
        args,
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_dab=args.use_dab,
        num_patterns=args.num_patterns,
        random_refpoints_xy=args.random_refpoints_xy,
        no_obj=args.no_obj
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_hoi_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_hoi_giou'] = args.giou_loss_coef
    # weight_dict['loss_hoi_score_ce'] = args.hoi_score_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
        
    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                no_obj=args.no_obj)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id, no_obj=args.no_obj)}

    return model, criterion, postprocessors
