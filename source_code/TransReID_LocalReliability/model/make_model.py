import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class VisibilityPatchWeighting(nn.Module):
    def __init__(self, in_planes, hidden_dim, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax
        self.weight_head = nn.Sequential(
            nn.Linear(in_planes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.weight_head.apply(weights_init_kaiming)

    def forward(self, patch_tokens):
        patch_logits = self.weight_head(patch_tokens).squeeze(-1)
        if self.use_softmax:
            patch_weights = F.softmax(patch_logits, dim=1)
        else:
            patch_weights = torch.sigmoid(patch_logits)
            patch_weights = patch_weights / (patch_weights.sum(dim=1, keepdim=True) + 1e-12)
        weighted_feat = torch.sum(patch_weights.unsqueeze(-1) * patch_tokens, dim=1)
        return weighted_feat, patch_weights, patch_logits


class ForegroundAwarePatchFiltering(nn.Module):
    def __init__(self, in_planes, hidden_dim):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(in_planes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.score_head.apply(weights_init_kaiming)

    def forward(self, patch_tokens, keep_ratio):
        patch_logits = self.score_head(patch_tokens).squeeze(-1)
        foreground_scores = torch.sigmoid(patch_logits)

        patch_count = patch_tokens.size(1)
        keep_count = max(1, int(patch_count * keep_ratio))
        keep_count = min(keep_count, patch_count)

        if keep_count == patch_count:
            keep_mask = torch.ones_like(foreground_scores)
        else:
            topk_scores, _ = torch.topk(foreground_scores, k=keep_count, dim=1)
            keep_threshold = topk_scores[:, -1:].detach()
            keep_mask = (foreground_scores >= keep_threshold).float()

        filtered_scores = foreground_scores * keep_mask
        filtered_weights = filtered_scores / (filtered_scores.sum(dim=1, keepdim=True) + 1e-12)
        filtered_feat = torch.sum(filtered_weights.unsqueeze(-1) * patch_tokens, dim=1)
        return filtered_feat, filtered_weights, foreground_scores, keep_mask


class ResidualVisibilityFusion(nn.Module):
    def __init__(self, in_planes, hidden_dim, use_softmax=True, gate_init=-2.0):
        super().__init__()
        self.visibility_weighting = VisibilityPatchWeighting(in_planes, hidden_dim, use_softmax)
        self.gate_head = nn.Sequential(
            nn.Linear(in_planes * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate_head.apply(weights_init_kaiming)
        nn.init.constant_(self.gate_head[-1].weight, 0.0)
        nn.init.constant_(self.gate_head[-1].bias, gate_init)

    def forward(self, base_global_feat, patch_tokens):
        weighted_feat, patch_weights, patch_logits = self.visibility_weighting(patch_tokens)
        gate_input = torch.cat(
            [base_global_feat, weighted_feat, torch.abs(base_global_feat - weighted_feat)], dim=1
        )
        residual_gate = torch.sigmoid(self.gate_head(gate_input))
        fused_feat = base_global_feat + residual_gate * (weighted_feat - base_global_feat)
        return fused_feat, weighted_feat, patch_weights, patch_logits, residual_gate.squeeze(-1)


class PatchReliabilityModeling(nn.Module):
    def __init__(self, in_planes, hidden_dim, use_softmax=False, fusion_alpha=0.25, gate_init=-2.0):
        super().__init__()
        self.use_softmax = use_softmax
        self.fusion_alpha = fusion_alpha
        self.reliability_head = nn.Sequential(
            nn.Linear(in_planes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.reliability_head.apply(weights_init_kaiming)
        self.gate_head = nn.Sequential(
            nn.Linear(in_planes * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate_head.apply(weights_init_kaiming)
        nn.init.constant_(self.gate_head[-1].weight, 0.0)
        nn.init.constant_(self.gate_head[-1].bias, gate_init)

    def forward(self, base_global_feat, patch_tokens):
        reliability_logits = self.reliability_head(patch_tokens).squeeze(-1)
        reliability_scores = torch.sigmoid(reliability_logits)
        if self.use_softmax:
            reliability_weights = F.softmax(reliability_logits, dim=1)
        else:
            # Compute reliability normalization in fp32 to avoid AMP/fp16
            # underflow when one-sided supervision drives most scores close to 0.
            reliability_scores_fp32 = reliability_scores.float().clamp(min=1e-4, max=1.0)
            reliability_weights = reliability_scores_fp32 / (
                reliability_scores_fp32.sum(dim=1, keepdim=True) + 1e-6
            )
            reliability_weights = reliability_weights.to(patch_tokens.dtype)

        reliability_feat = torch.sum(reliability_weights.unsqueeze(-1) * patch_tokens.float(), dim=1)
        reliability_feat = reliability_feat.to(patch_tokens.dtype)
        gate_input = torch.cat(
            [base_global_feat, reliability_feat, torch.abs(base_global_feat - reliability_feat)], dim=1
        )
        reliability_gate = torch.sigmoid(self.gate_head(gate_input))
        fused_feat = base_global_feat + self.fusion_alpha * reliability_gate * (reliability_feat - base_global_feat)

        return fused_feat, {
            'base_global_feat': base_global_feat,
            'fused_global_feat': fused_feat,
            'patch_tokens': patch_tokens,
            'reliability_logits': reliability_logits,
            'patch_reliability': reliability_scores,
            'patch_weights': reliability_weights,
            'reliability_feat': reliability_feat,
            'reliability_gate': reliability_gate.squeeze(-1),
        }


def fuse_global_and_visibility_features(global_feat, weighted_feat, alpha):
    return alpha * global_feat + (1.0 - alpha) * weighted_feat


def aggregate_patch_weights_to_local_branches(patch_weights, divide_length, rearrange, shift_num, shuffle_groups):
    local_weight_tokens = patch_weights.unsqueeze(-1)
    if rearrange:
        local_weight_tokens = shuffle_unit(local_weight_tokens, shift_num, shuffle_groups)

    patch_count = local_weight_tokens.size(1)
    patch_length = patch_count // divide_length
    branch_weights = []
    for branch_idx in range(divide_length):
        start = branch_idx * patch_length
        end = patch_count if branch_idx == divide_length - 1 else (branch_idx + 1) * patch_length
        branch_weight = local_weight_tokens[:, start:end, 0].mean(dim=1)
        branch_weights.append(branch_weight)

    local_branch_weights = torch.stack(branch_weights, dim=1)
    local_branch_weights = local_branch_weights / (local_branch_weights.sum(dim=1, keepdim=True) + 1e-6)
    return local_branch_weights


def resolve_pretrained_path(cfg, model_path):
    if model_path and os.path.exists(model_path):
        return model_path
    # Only reuse TEST.WEIGHT as a fallback for whole-model loading. When the
    # config asks for ImageNet initialization, TEST.WEIGHT is a downstream
    # checkpoint with different key names (e.g. base.cls_token) and cannot be
    # loaded directly into the backbone.
    if cfg.MODEL.PRETRAIN_CHOICE in ('self', 'finetune') and cfg.TEST.WEIGHT and os.path.exists(cfg.TEST.WEIGHT):
        return cfg.TEST.WEIGHT
    return model_path


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = resolve_pretrained_path(cfg, cfg.MODEL.PRETRAIN_PATH)
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_path = resolve_pretrained_path(cfg, model_path)
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_reliability_pipeline = cfg.MODEL.RELIABILITY_PIPELINE
        self.use_vis_weighting = cfg.MODEL.VIS_WEIGHTING
        self.vis_mode = cfg.MODEL.VIS_MODE
        self.vis_fusion_alpha = cfg.MODEL.VIS_FUSION_ALPHA
        self.fg_keep_ratio = cfg.MODEL.FG_KEEP_RATIO
        self.vis_gate_init = cfg.MODEL.VIS_GATE_INIT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        self.patch_grid = (self.base.patch_embed.num_y, self.base.patch_embed.num_x)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.base_bottleneck = None
        self.base_classifier = None

        if self.use_reliability_pipeline:
            self.patch_reliability_modeling = PatchReliabilityModeling(
                self.in_planes,
                cfg.MODEL.REL_HIDDEN_DIM,
                cfg.MODEL.REL_USE_SOFTMAX,
                cfg.MODEL.REL_FUSION_ALPHA,
                cfg.MODEL.REL_GATE_INIT
            )
            print('using reliability-aware patch modeling on local JPM branch')
        elif self.use_vis_weighting and self.vis_mode == 'global_fusion':
            self.visibility_weighting = VisibilityPatchWeighting(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM, cfg.MODEL.VIS_USE_SOFTMAX
            )
            print('using visibility-guided patch weighting with fusion alpha:{}'.format(self.vis_fusion_alpha))
        elif self.use_vis_weighting and self.vis_mode == 'residual_fusion':
            self.residual_visibility_fusion = ResidualVisibilityFusion(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM, cfg.MODEL.VIS_USE_SOFTMAX, self.vis_gate_init
            )
            self.base_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.base_bottleneck.bias.requires_grad_(False)
            self.base_bottleneck.apply(weights_init_kaiming)
            self.base_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.base_classifier.apply(weights_init_classifier)
            print('using residual visibility-guided fusion with gate init:{}'.format(self.vis_gate_init))
        elif self.use_vis_weighting and self.vis_mode == 'foreground_filter':
            self.foreground_filter = ForegroundAwarePatchFiltering(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM
            )
            print('using foreground-aware patch filtering with keep ratio:{} and fusion alpha:{}'.format(
                self.fg_keep_ratio, self.vis_fusion_alpha
            ))
        if pretrain_choice in ('self', 'finetune'):
            self.load_param(model_path)
            print('Loading pretrained whole model......from {}'.format(model_path))

    def forward(self, x, label=None, cam_label= None, view_label=None):
        aux_outputs = None
        if self.use_reliability_pipeline:
            tokens = self.base(x, cam_label=cam_label, view_label=view_label, return_all_tokens=True)
            base_global_feat = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
            global_feat, aux_outputs = self.patch_reliability_modeling(base_global_feat, patch_tokens)
            aux_outputs['patch_grid'] = self.patch_grid
        elif self.use_vis_weighting and self.vis_mode in ('global_fusion', 'foreground_filter', 'residual_fusion'):
            tokens = self.base(x, cam_label=cam_label, view_label=view_label, return_all_tokens=True)
            base_global_feat = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
            if self.vis_mode == 'global_fusion':
                weighted_feat, patch_weights, patch_logits = self.visibility_weighting(patch_tokens)
                global_feat = fuse_global_and_visibility_features(
                    base_global_feat, weighted_feat, self.vis_fusion_alpha
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'weighted_patch_feat': weighted_feat,
                    'patch_weights': patch_weights,
                    'patch_logits': patch_logits,
                }
            elif self.vis_mode == 'residual_fusion':
                global_feat, weighted_feat, patch_weights, patch_logits, residual_gate = self.residual_visibility_fusion(
                    base_global_feat, patch_tokens
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'weighted_patch_feat': weighted_feat,
                    'fused_global_feat': global_feat,
                    'patch_weights': patch_weights,
                    'patch_logits': patch_logits,
                    'residual_gate': residual_gate,
                }
            else:
                filtered_feat, patch_weights, foreground_scores, keep_mask = self.foreground_filter(
                    patch_tokens, self.fg_keep_ratio
                )
                global_feat = fuse_global_and_visibility_features(
                    base_global_feat, filtered_feat, self.vis_fusion_alpha
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'filtered_patch_feat': filtered_feat,
                    'patch_weights': patch_weights,
                    'foreground_scores': foreground_scores,
                    'foreground_keep_mask': keep_mask,
                }
        else:
            global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                base_score = self.base_classifier(self.base_bottleneck(base_global_feat), label) if self.base_classifier is not None else None
            else:
                cls_score = self.classifier(feat)
                base_score = self.base_classifier(self.base_bottleneck(base_global_feat)) if self.base_classifier is not None else None
            if base_score is not None:
                return [cls_score, base_score], [global_feat, base_global_feat], aux_outputs
            return cls_score, global_feat, aux_outputs  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        state_dict = self.state_dict()
        for i in param_dict:
            key = i.replace('module.', '')
            if key in state_dict and state_dict[key].shape == param_dict[i].shape:
                state_dict[key].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        state_dict = self.state_dict()
        for i in param_dict:
            if i in state_dict and state_dict[i].shape == param_dict[i].shape:
                state_dict[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_path = resolve_pretrained_path(cfg, model_path)
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_reliability_pipeline = cfg.MODEL.RELIABILITY_PIPELINE
        self.use_vis_weighting = cfg.MODEL.VIS_WEIGHTING
        self.vis_mode = cfg.MODEL.VIS_MODE
        self.vis_fusion_alpha = cfg.MODEL.VIS_FUSION_ALPHA
        self.fg_keep_ratio = cfg.MODEL.FG_KEEP_RATIO
        self.vis_gate_init = cfg.MODEL.VIS_GATE_INIT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.patch_grid = (self.base.patch_embed.num_y, self.base.patch_embed.num_x)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.base_bottleneck = None
        self.base_classifier = None
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

        if self.use_reliability_pipeline:
            self.patch_reliability_modeling = PatchReliabilityModeling(
                self.in_planes,
                cfg.MODEL.REL_HIDDEN_DIM,
                cfg.MODEL.REL_USE_SOFTMAX,
                cfg.MODEL.REL_FUSION_ALPHA,
                cfg.MODEL.REL_GATE_INIT
            )
            print('using reliability-aware patch modeling with fusion alpha:{}'.format(cfg.MODEL.REL_FUSION_ALPHA))
        elif self.use_vis_weighting and self.vis_mode == 'global_fusion':
            self.visibility_weighting = VisibilityPatchWeighting(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM, cfg.MODEL.VIS_USE_SOFTMAX
            )
            print('using visibility-guided patch weighting with fusion alpha:{}'.format(self.vis_fusion_alpha))
        elif self.use_vis_weighting and self.vis_mode == 'residual_fusion':
            self.residual_visibility_fusion = ResidualVisibilityFusion(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM, cfg.MODEL.VIS_USE_SOFTMAX, self.vis_gate_init
            )
            self.base_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.base_bottleneck.bias.requires_grad_(False)
            self.base_bottleneck.apply(weights_init_kaiming)
            self.base_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.base_classifier.apply(weights_init_classifier)
            print('using residual visibility-guided fusion with gate init:{}'.format(self.vis_gate_init))
        elif self.use_vis_weighting and self.vis_mode == 'foreground_filter':
            self.foreground_filter = ForegroundAwarePatchFiltering(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM
            )
            print('using foreground-aware patch filtering with keep ratio:{} and fusion alpha:{}'.format(
                self.fg_keep_ratio, self.vis_fusion_alpha
            ))
        elif self.use_vis_weighting and self.vis_mode == 'local_jpm':
            self.local_branch_weighting = VisibilityPatchWeighting(
                self.in_planes, cfg.MODEL.VIS_HIDDEN_DIM, cfg.MODEL.VIS_USE_SOFTMAX
            )
            print('using visibility-guided local branch weighting on JPM')
        if pretrain_choice in ('self', 'finetune'):
            self.load_param(model_path)
            print('Loading pretrained whole model......from {}'.format(model_path))

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        aux_outputs = None

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        base_global_feat = b1_feat[:, 0]
        global_feat = base_global_feat
        if self.use_reliability_pipeline:
            patch_tokens = b1_feat[:, 1:]
            _, aux_outputs = self.patch_reliability_modeling(base_global_feat, patch_tokens)
            aux_outputs['patch_grid'] = self.patch_grid
        elif self.use_vis_weighting and self.vis_mode in ('global_fusion', 'foreground_filter', 'residual_fusion'):
            patch_tokens = b1_feat[:, 1:]
            if self.vis_mode == 'global_fusion':
                weighted_feat, patch_weights, patch_logits = self.visibility_weighting(patch_tokens)
                global_feat = fuse_global_and_visibility_features(
                    base_global_feat, weighted_feat, self.vis_fusion_alpha
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'weighted_patch_feat': weighted_feat,
                    'patch_weights': patch_weights,
                    'patch_logits': patch_logits,
                }
            elif self.vis_mode == 'residual_fusion':
                global_feat, weighted_feat, patch_weights, patch_logits, residual_gate = self.residual_visibility_fusion(
                    base_global_feat, patch_tokens
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'weighted_patch_feat': weighted_feat,
                    'fused_global_feat': global_feat,
                    'patch_weights': patch_weights,
                    'patch_logits': patch_logits,
                    'residual_gate': residual_gate,
                }
            else:
                filtered_feat, patch_weights, foreground_scores, keep_mask = self.foreground_filter(
                    patch_tokens, self.fg_keep_ratio
                )
                global_feat = fuse_global_and_visibility_features(
                    base_global_feat, filtered_feat, self.vis_fusion_alpha
                )
                aux_outputs = {
                    'base_global_feat': base_global_feat,
                    'filtered_patch_feat': filtered_feat,
                    'patch_weights': patch_weights,
                    'foreground_scores': foreground_scores,
                    'foreground_keep_mask': keep_mask,
                }

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        if self.use_reliability_pipeline and aux_outputs is not None:
            local_weights = aggregate_patch_weights_to_local_branches(
                aux_outputs['patch_weights'],
                self.divide_length,
                self.rearrange,
                self.shift_num,
                self.shuffle_groups
            )
            # Preserve the original JPM feature scale: average weight is 1 / divide_length.
            local_scale = local_weights * self.divide_length
            local_feat_1 = local_feat_1 * local_scale[:, 0:1]
            local_feat_2 = local_feat_2 * local_scale[:, 1:2]
            local_feat_3 = local_feat_3 * local_scale[:, 2:3]
            local_feat_4 = local_feat_4 * local_scale[:, 3:4]
            aux_outputs['local_weights'] = local_weights
            aux_outputs['local_scale'] = local_scale
        elif self.use_vis_weighting and self.vis_mode == 'local_jpm':
            local_feat_stack = torch.stack(
                [local_feat_1, local_feat_2, local_feat_3, local_feat_4], dim=1
            )
            _, local_weights, local_logits = self.local_branch_weighting(local_feat_stack)
            local_feat_1 = local_feat_1 * local_weights[:, 0:1]
            local_feat_2 = local_feat_2 * local_weights[:, 1:2]
            local_feat_3 = local_feat_3 * local_weights[:, 2:3]
            local_feat_4 = local_feat_4 * local_weights[:, 3:4]
            aux_outputs = {
                'local_weights': local_weights,
                'local_logits': local_logits,
            }

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                cls_score_base = self.base_classifier(self.base_bottleneck(base_global_feat), label) if self.base_classifier is not None else None
            else:
                cls_score = self.classifier(feat)
                cls_score_base = self.base_classifier(self.base_bottleneck(base_global_feat)) if self.base_classifier is not None else None
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            score_outputs = [cls_score]
            feat_outputs = [global_feat]
            if cls_score_base is not None:
                score_outputs.append(cls_score_base)
                feat_outputs.append(base_global_feat)
            score_outputs.extend([cls_score_1, cls_score_2, cls_score_3, cls_score_4])
            feat_outputs.extend([local_feat_1, local_feat_2, local_feat_3, local_feat_4])
            return score_outputs, feat_outputs, aux_outputs  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        state_dict = self.state_dict()
        for i in param_dict:
            key = i.replace('module.', '')
            if key in state_dict and state_dict[key].shape == param_dict[i].shape:
                state_dict[key].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        state_dict = self.state_dict()
        for i in param_dict:
            if i in state_dict and state_dict[i].shape == param_dict[i].shape:
                state_dict[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
