
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import einops
from ..functions import MSDeformAttnFunction
import time
# from attention import MultiheadAttention
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max
    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6
class MSDeformAttn(nn.Module):
    def __init__(self, arg, process='Decoder', d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = 8
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.scale = arg.scale #[1, 3, 5]
        self.merge = arg.merge #True
        self.spatial_merge = arg.hierarchical_merge  #False
        self.scale_merge = arg.hierarchical_merge
        self.task_merge = False
        self.task_mergev2 = arg.task_merge
        self.base = arg.base
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.update = False
        if process == 'Decoder' and self.merge:
            if self.spatial_merge == True:
                self.multi_attn_spatial = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
                self.q_spital = nn.Linear(d_model, d_model, bias=True)
                self.kv_spital = nn.Linear(d_model, 2*d_model, bias=True)
                self.norm_spatial = nn.LayerNorm(d_model)
            if self.scale_merge == True:
                self.multi_attn_scale = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
                self.q_scale = nn.Linear(d_model, d_model, bias=True)
                self.kv_scale = nn.Linear(d_model, 2*d_model, bias=True)
                self.norm_scale = nn.LayerNorm(d_model)
        # self.softmax = nn.Softmax(dim=-1)
        # self.attn_drop = nn.Dropout(0)
        # self.proj_spital = nn.Linear(d_model, d_model)
        # self.proj_scale = nn.Linear(d_model, d_model)
        # self.proj_drop = nn.Dropout(0)
            if self.task_merge == True:
                self.fc = nn.Sequential(
                    nn.Linear(d_model*2, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model*4)
                )
            if self.task_mergev2 == True:
                self.multi_attn_task = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
                self.q_task = nn.Linear(d_model, d_model, bias=True)
                self.kv_task = nn.Linear(d_model, 2*d_model, bias=True)
                self.norm_task = nn.LayerNorm(d_model)
                self.fcV2 = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model*4)
                )

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def generate_points(self, H, W, scale, points):
        h_scale, w_scale = 2/H, 2/W
        generate_points = points[:, :, None, None, :].repeat(1, 1, scale, scale, 1)
        Midpoint = scale // 2 
        for i in range(scale):
            for j in range(scale):
                generate_points[:,:,i,j,0] = points[:,:,0] + (i-Midpoint)*h_scale                 
                generate_points[:,:,i,j,1] = points[:,:,1] + (j-Midpoint)*w_scale
        return einops.rearrange(generate_points, 'B N H W P -> B N (H W) P')
    def Sampling(self, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        # input_spatial_shapes H W
        # anchor to sample features
        sample_features = []
        B, _, dim = input_flatten.shape
        for i in range(len(input_level_start_index)):
            if i == len(input_level_start_index)-1:
                feature = einops.rearrange(input_flatten[:, input_level_start_index[i]:, :]. \
                    view(B, input_spatial_shapes[i][0], input_spatial_shapes[i][1], dim), 'B H W C -> B C H W')
                sample_points = 2 * reference_points[:, :, i, :] - 1
                sample_points = self.generate_points(input_spatial_shapes[i][0], input_spatial_shapes[i][1], self.scale[i], sample_points)
                sample = F.grid_sample(
                                    input=feature, 
                                    grid=sample_points,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
                sample_features.append(einops.rearrange(sample, 'B C N (H W) -> B N (H W) C', H=self.scale[i])) 
                # sample_features.append(sample)
            else:
                feature = einops.rearrange(input_flatten[:, input_level_start_index[i]:input_level_start_index[i+1], :]. \
                    view(B, input_spatial_shapes[i][0], input_spatial_shapes[i][1], dim), 'B H W C -> B C H W')
                sample_points = 2 * reference_points[:, :, i, :] - 1
                sample_points = self.generate_points(input_spatial_shapes[i][0], input_spatial_shapes[i][1], self.scale[i], sample_points)
                sample = F.grid_sample(
                                    input=feature, 
                                    grid=sample_points,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
                sample_features.append(einops.rearrange(sample, 'B C N (H W) -> B N (H W) C', H=self.scale[i])) 
                # sample_features.append(sample)
        # sample_feature = torch.cat(sample_features, dim=1)
        # sample_feature = torch.mean(sample_feature, dim=1, keepdim=True).squeeze(1)
        return sample_features
    def ROI(self, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        # input_spatial_shapes H W
        # anchor to sample features
        sample_features = []
        B, _, dim = input_flatten.shape
        for i in range(len(input_level_start_index)):
            if i == len(input_level_start_index)-1:
                feature = einops.rearrange(input_flatten[:, input_level_start_index[i]:, :]. \
                    view(B, input_spatial_shapes[i][0], input_spatial_shapes[i][1], dim), 'B H W C -> B C H W')
                sample_points = 2 * reference_points[:, :, i, :] - 1
                sample = F.grid_sample(
                                    input=feature, 
                                    grid=sample_points[:,:,None,:],
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
                sample_features.append(einops.rearrange(sample, 'B C N H -> B H N C')) 
            else:
                feature = einops.rearrange(input_flatten[:, input_level_start_index[i]:input_level_start_index[i+1], :]. \
                    view(B, input_spatial_shapes[i][0], input_spatial_shapes[i][1], dim), 'B H W C -> B C H W')
                sample_points = 2 * reference_points[:, :, i, :] - 1
                sample = F.grid_sample(
                                    input=feature, 
                                    grid=sample_points[:,:,None,:],
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
                sample_features.append(einops.rearrange(sample, 'B C N H -> B H N C'))
        sample_feature = torch.cat(sample_features, dim=1)
        sample_feature = torch.mean(sample_feature, dim=1, keepdim=True).squeeze(1)
        return sample_feature
    def base_set(self, query, sample_features):
        for i in range(len(sample_features)):
            query = query + torch.mean(sample_features[i], dim=-2).squeeze(-2)/len(sample_features)
        return query
    def merge_attention(self, query, sample_feature):
        B, N, P, C = sample_feature.shape
        q = self.q(query[:,:,None,:]).reshape(B, N, 1, self.num_head, self.d_model//self.num_head).permute(0, 1, 3, 2, 4)
        kv = self.kv(sample_feature).reshape(B, N, P, 2, self.num_head, self.d_model//self.num_head).permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]
        q = q * (32 ** -0.5)
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        merge_feature = (attn @ v).transpose(1, 2).reshape(B, N, C)
        merge_feature = self.proj(merge_feature)
        merge_feature = self.proj_drop(merge_feature)
        return merge_feature
    # multi-head hen zhong yao 
    def Spatial_aware(self, query, sample_features):
        merge_features = []
        for i in range(len(sample_features)):
            B, N, P, C = sample_features[i].shape
            q = self.q_spital(query[:,:,None,:]).reshape(B, N, 1, C)
            kv = self.kv_spital(sample_features[i]).reshape(B, N, P, 2, C).permute(3, 0, 1, 2, 4)
            # q = self.q_spital(query[:,:,None,:]).reshape(B, N, 1, self.num_head, self.d_model//self.num_head).permute(0, 1, 3, 2, 4)
            # kv = self.kv_spital(sample_features[i]).reshape(B, N, P, 2, self.num_head, self.d_model//self.num_head).permute(3, 0, 1, 4, 2, 5)
            k, v = kv[0], kv[1]
            q = einops.rearrange(q, 'B N L C -> (B N) L C')
            k = einops.rearrange(k, 'B N L C -> (B N) L C')
            v = einops.rearrange(v, 'B N L C -> (B N) L C')
            merge_feature = self.multi_attn_spatial(q, k, v)[0]
            merge_feature = einops.rearrange(merge_feature, '(B N) L C -> B N L C',B=B)
            merge_features.append(merge_feature)
        return merge_features
    def base_spatial(self, sample_features):
        merge_features = [torch.mean(x, dim=-2).unsqueeze(-2) for x in sample_features]
        return merge_features
    # multi-head hen zhong yao 
    def Scale_aware(self, query, merge_features):
        sample_feature = torch.cat(merge_features, dim=-2)
        B, N, P, C = sample_feature.shape
        q = self.q_scale(query[:,:,None,:]).reshape(B, N, 1, C)
        kv = self.kv_scale(sample_feature).reshape(B, N, P, 2, C).permute(3, 0, 1, 2, 4)
        # q = self.q_scale(query[:,:,None,:]).reshape(B, N, 1, self.num_head, self.d_model//self.num_head).permute(0, 1, 3, 2, 4)
        # kv = self.kv_scale(sample_feature).reshape(B, N, P, 2, self.num_head, self.d_model//self.num_head).permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]
        q = einops.rearrange(q, 'B N L C -> (B N) L C')
        k = einops.rearrange(k, 'B N L C -> (B N) L C')
        v = einops.rearrange(v, 'B N L C -> (B N) L C')
        merge_feature = self.multi_attn_scale(q, k, v)[0]
        merge_feature = einops.rearrange(merge_feature, '(B N) L C -> B N L C',B=B)
        # q = q * (256 ** -0.5)
        # q = q * (32 ** -0.5)
        # attn = (q @ k.transpose(-2, -1))
        # attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        # merge_feature = (attn @ v).transpose(-3, -2).reshape(B, N, 1, C)
        # merge_feature = self.proj_scale(merge_feature)
        # merge_feature = self.proj_drop(merge_feature)
        return merge_feature.squeeze(-2)
    def base_scale(self, merge_features):
        return torch.mean(torch.cat(merge_features, dim=-2), dim=-2)
    def Task_aware(self, query, update_query):
        b, n, c = query.size()
        update_query = update_query.squeeze(-2)
        # sum_query = torch.cat([query + update_query, query, update_query], dim=-1)
        sum_query = torch.cat([query + update_query, query], dim=-1)
        # sum_query = torch.cat([query, update_query], dim=-1)
        param = F.normalize(self.fc(sum_query).view(b, n, 4*c))
        a1, b1, a2, b2 = torch.split(param, c, dim=2)
        a1 = (a1 - 0.5) * 2 + 1  # 1.0
        a2 = (a2 - 0.5) * 2 + 0
        b1 = b1 - 0.5 + 0
        b2 = b2 - 0.5 + 0
        update_query2 = torch.max(update_query * a1 + b1, update_query * a2 + b2)
        query = query + update_query2
        return query
    def Task_awareV2(self, query, update_query):
        sum_query = torch.cat([query[:, :, None, :], update_query[:, :, None, :]], dim=-2) #b n 2 c
        B, N, P, C = sum_query.shape
        q = self.q_task(query[:,:,None,:]).reshape(B, N, 1, C)
        kv = self.kv_task(sum_query).reshape(B, N, P, 2, C).permute(3, 0, 1, 2, 4)
        k, v = kv[0], kv[1]
        q = einops.rearrange(q, 'B N L C -> (B N) L C')
        k = einops.rearrange(k, 'B N L C -> (B N) L C')
        v = einops.rearrange(v, 'B N L C -> (B N) L C')
        merge_query = self.multi_attn_task(q, k, v)[0]
        merge_query = einops.rearrange(merge_query, '(B N) L C -> B N L C',B=B).squeeze(-2)
        param = F.normalize(self.fcV2(merge_query).view(B, N, 4*C))
        a1, b1, a2, b2 = torch.split(param, C, dim=2)
        a1 = (a1 - 0.5) * 2 + 1  # 1.0
        a2 = (a2 - 0.5) * 2 + 0
        b1 = b1 - 0.5 + 0
        b2 = b2 - 0.5 + 0
        update_query2 = torch.max(update_query * a1 + b1, update_query * a2 + b2)
        query = query + update_query2
        return query
    def base_task(self, query, update_query):
        return query + update_query
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        process_name = query['process']
        query = query['query']
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if process_name == 'Decoder' and self.merge == True:
            sample_features = self.Sampling(reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
            if self.base:
                query = self.base_set(query, sample_features)      # 2*300*256
            else:
                if self.spatial_merge:
                    merge_features = self.Spatial_aware(query, sample_features)    #3 2*300*1*256
                else:
                    merge_features = self.base_spatial(sample_features)   #3 2*300*1*256
                if self.scale_merge:
                    update_query = self.Scale_aware(query, merge_features) # 2*300*256
                else:
                    update_query = self.base_scale(merge_features)   # 2*300*256
                if self.task_merge:
                    query = self.Task_aware(query, update_query)    #2*300*256
                elif self.task_mergev2:
                    query = self.Task_awareV2(query, update_query) 
                else:
                    query = self.base_task(query, update_query)     #2*300*256
         
            # update_query = self.ROI(reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
            # query = query + update_query
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # None xiang dang yu unsqueeze de zuo yong
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # if process_name == 'Decoder' and self.update:
        #     B, N, H, L, S = attention_weights.shape
        #     weights = einops.rearrange(attention_weights, 'B N H L S -> (B N ) (L H S)')
        #     update_points = einops.rearrange(sampling_locations, 'B N H L S P -> (B N ) (L H S) P')
        #     max_index = torch.argmax(weights, dim=-1, keepdim=False)
        #     update_reference_points = torch.zeros_like(max_index, dtype=torch.float32, device=attention_weights.device).unsqueeze(-1).repeat(1,2) # 1800 1
        #     for i in range(len(max_index)):
        #         update_reference_points[i] = update_points[i,max_index[i],:]
        #     reference_points = einops.rearrange(update_reference_points, '(B N) P -> B N P', B=B, N=N)
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output