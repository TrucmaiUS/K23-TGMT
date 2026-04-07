import torch
import torch.nn as nn
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


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
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
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.enable_patch_weight = cfg.MODEL.VIS_WEIGHT.ENABLED
        self.enable_local_group = cfg.MODEL.LOCAL_GROUP.ENABLED
        self.token_branch_enabled = self.enable_patch_weight or self.enable_local_group
        self.patch_weight_eps = cfg.MODEL.VIS_WEIGHT.EPS
        self.patch_weight_warmup_epochs = cfg.MODEL.VIS_WEIGHT.WARMUP_EPOCHS
        self.local_row_bounds = list(cfg.MODEL.LOCAL_GROUP.ROW_BOUNDS)

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

        if self.enable_patch_weight:
            hidden_dim = max(1, self.in_planes // cfg.MODEL.VIS_WEIGHT.HIDDEN_DIM_RATIO)
            self.patch_weight_head = nn.Sequential(
                nn.Linear(self.in_planes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.patch_weight_head.apply(weights_init_kaiming)
        else:
            self.patch_weight_head = None

        if self.enable_local_group:
            if len(self.local_row_bounds) != 4:
                raise ValueError('MODEL.LOCAL_GROUP.ROW_BOUNDS must contain 4 values for 3 regions.')
            if self.local_row_bounds[0] != 0 or self.local_row_bounds[-1] != self.patch_grid[0]:
                raise ValueError(
                    'Local grouping row bounds {} do not match patch grid height {}.'.format(
                        self.local_row_bounds, self.patch_grid[0]
                    )
                )
            self.bottleneck_up = self._build_bnneck()
            self.bottleneck_mid = self._build_bnneck()
            self.bottleneck_low = self._build_bnneck()
            self.classifier_up = self._build_linear_classifier()
            self.classifier_mid = self._build_linear_classifier()
            self.classifier_low = self._build_linear_classifier()

    def _build_bnneck(self):
        bottleneck = nn.BatchNorm1d(self.in_planes)
        bottleneck.bias.requires_grad_(False)
        bottleneck.apply(weights_init_kaiming)
        return bottleneck

    def _build_linear_classifier(self):
        classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        classifier.apply(weights_init_classifier)
        return classifier

    def _apply_global_classifier(self, feat, label):
        if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            return self.classifier(feat, label)
        return self.classifier(feat)

    def _get_patch_weights(self, patch_tokens, epoch):
        if not self.enable_patch_weight:
            return None

        if self.training and epoch is not None and epoch <= self.patch_weight_warmup_epochs:
            return torch.ones(
                patch_tokens.size(0), patch_tokens.size(1), 1,
                device=patch_tokens.device, dtype=patch_tokens.dtype
            )

        return self.patch_weight_head(patch_tokens)

    def _pool_local_regions(self, patch_tokens):
        batch, _, dim = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(batch, self.patch_grid[0], self.patch_grid[1], dim)
        local_feats = []
        for start_row, end_row in zip(self.local_row_bounds[:-1], self.local_row_bounds[1:]):
            region_tokens = patch_tokens[:, start_row:end_row, :, :].reshape(batch, -1, dim)
            local_feats.append(region_tokens.mean(dim=1))
        return local_feats

    def forward(self, x, label=None, cam_label= None, view_label=None, epoch=None, return_feature_dict=False):
        if not self.token_branch_enabled:
            global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
            feat = self.bottleneck(global_feat)

            if self.training:
                cls_score = self._apply_global_classifier(feat, label)
                return cls_score, global_feat

            if self.neck_feat == 'after':
                return feat
            return global_feat

        token_features = self.base(x, cam_label=cam_label, view_label=view_label, return_all_tokens=True)
        cls_token = token_features[:, 0]
        patch_tokens = token_features[:, 1:]
        patch_weights = self._get_patch_weights(patch_tokens, epoch)

        if patch_weights is not None:
            weighted_tokens = patch_weights * patch_tokens
            global_feat = weighted_tokens.sum(dim=1) / (patch_weights.sum(dim=1) + self.patch_weight_eps)
        else:
            global_feat = cls_token

        global_bn_feat = self.bottleneck(global_feat)
        local_feats = []
        local_bn_feats = []
        local_logits = []

        if self.enable_local_group:
            local_feats = self._pool_local_regions(patch_tokens)
            local_bn_feats = [
                self.bottleneck_up(local_feats[0]),
                self.bottleneck_mid(local_feats[1]),
                self.bottleneck_low(local_feats[2])
            ]
            if self.training:
                local_logits = [
                    self.classifier_up(local_bn_feats[0]),
                    self.classifier_mid(local_bn_feats[1]),
                    self.classifier_low(local_bn_feats[2])
                ]

        if self.training:
            outputs = {
                "global_feat": global_feat,
                "global_bn_feat": global_bn_feat,
                "global_logits": self._apply_global_classifier(global_bn_feat, label),
                "local_feats": local_feats,
                "local_bn_feats": local_bn_feats,
                "local_logits": local_logits,
                "patch_weights": patch_weights.squeeze(-1) if patch_weights is not None else None,
                "patch_tokens": patch_tokens,
            }
            return outputs

        retrieval_feat = global_bn_feat if self.neck_feat == 'after' else global_feat
        if return_feature_dict:
            return {
                "retrieval_feat": retrieval_feat,
                "global_feat": global_feat,
                "global_bn_feat": global_bn_feat,
                "local_feats": local_feats,
                "local_bn_feats": local_bn_feats,
                "patch_weights": patch_weights.squeeze(-1) if patch_weights is not None else None,
                "patch_tokens": patch_tokens,
            }
        return retrieval_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.enable_patch_weight = cfg.MODEL.VIS_WEIGHT.ENABLED
        self.enable_local_group = cfg.MODEL.LOCAL_GROUP.ENABLED
        self.token_branch_enabled = self.enable_patch_weight or self.enable_local_group
        self.patch_weight_eps = cfg.MODEL.VIS_WEIGHT.EPS
        self.patch_weight_warmup_epochs = cfg.MODEL.VIS_WEIGHT.WARMUP_EPOCHS
        self.local_row_bounds = list(cfg.MODEL.LOCAL_GROUP.ROW_BOUNDS)

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

        if self.enable_patch_weight:
            hidden_dim = max(1, self.in_planes // cfg.MODEL.VIS_WEIGHT.HIDDEN_DIM_RATIO)
            self.patch_weight_head = nn.Sequential(
                nn.Linear(self.in_planes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.patch_weight_head.apply(weights_init_kaiming)
        else:
            self.patch_weight_head = None

        if self.enable_local_group:
            if len(self.local_row_bounds) != 4:
                raise ValueError('MODEL.LOCAL_GROUP.ROW_BOUNDS must contain 4 values for 3 regions.')
            if self.local_row_bounds[0] != 0 or self.local_row_bounds[-1] != self.patch_grid[0]:
                raise ValueError(
                    'Local grouping row bounds {} do not match patch grid height {}.'.format(
                        self.local_row_bounds, self.patch_grid[0]
                    )
                )
            self.bottleneck_up = self._build_bnneck()
            self.bottleneck_mid = self._build_bnneck()
            self.bottleneck_low = self._build_bnneck()
            self.classifier_up = self._build_linear_classifier()
            self.classifier_mid = self._build_linear_classifier()
            self.classifier_low = self._build_linear_classifier()

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def _build_bnneck(self):
        bottleneck = nn.BatchNorm1d(self.in_planes)
        bottleneck.bias.requires_grad_(False)
        bottleneck.apply(weights_init_kaiming)
        return bottleneck

    def _build_linear_classifier(self):
        classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        classifier.apply(weights_init_classifier)
        return classifier

    def _get_patch_weights(self, patch_tokens, epoch):
        if not self.enable_patch_weight:
            return None

        if self.training and epoch is not None and epoch <= self.patch_weight_warmup_epochs:
            return torch.ones(
                patch_tokens.size(0), patch_tokens.size(1), 1,
                device=patch_tokens.device, dtype=patch_tokens.dtype
            )

        return self.patch_weight_head(patch_tokens)

    def _pool_local_regions(self, patch_tokens):
        batch, _, dim = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(batch, self.patch_grid[0], self.patch_grid[1], dim)
        local_feats = []
        for start_row, end_row in zip(self.local_row_bounds[:-1], self.local_row_bounds[1:]):
            region_tokens = patch_tokens[:, start_row:end_row, :, :].reshape(batch, -1, dim)
            local_feats.append(region_tokens.mean(dim=1))
        return local_feats

    def forward(self, x, label=None, cam_label= None, view_label=None, epoch=None, return_feature_dict=False):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]
        patch_tokens = b1_feat[:, 1:]
        patch_weights = self._get_patch_weights(patch_tokens, epoch)
        if patch_weights is not None:
            weighted_global_feat = (patch_weights * patch_tokens).sum(dim=1) / (patch_weights.sum(dim=1) + self.patch_weight_eps)
        else:
            weighted_global_feat = global_feat

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

        feat = self.bottleneck(weighted_global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        local_group_feats = []
        local_group_bn_feats = []
        local_group_logits = []

        if self.enable_local_group:
            local_group_feats = self._pool_local_regions(patch_tokens)
            local_group_bn_feats = [
                self.bottleneck_up(local_group_feats[0]),
                self.bottleneck_mid(local_group_feats[1]),
                self.bottleneck_low(local_group_feats[2])
            ]
            if self.training:
                local_group_logits = [
                    self.classifier_up(local_group_bn_feats[0]),
                    self.classifier_mid(local_group_bn_feats[1]),
                    self.classifier_low(local_group_bn_feats[2])
                ]

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                jpm_scores = cls_score
                jpm_feats = weighted_global_feat
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
                jpm_scores = [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4]
                jpm_feats = [weighted_global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]

            if self.token_branch_enabled:
                return {
                    "global_feat": weighted_global_feat,
                    "global_bn_feat": feat,
                    "global_logits": cls_score,
                    "local_feats": local_group_feats,
                    "local_bn_feats": local_group_bn_feats,
                    "local_logits": local_group_logits,
                    "patch_weights": patch_weights.squeeze(-1) if patch_weights is not None else None,
                    "patch_tokens": patch_tokens,
                    "base_scores": jpm_scores,
                    "base_feats": jpm_feats,
                    "jpm_local_feats": [local_feat_1, local_feat_2, local_feat_3, local_feat_4],
                    "jpm_local_bn_feats": [local_feat_1_bn, local_feat_2_bn, local_feat_3_bn, local_feat_4_bn],
                }

            return jpm_scores, jpm_feats  # global feature for triplet loss
        else:
            if self.token_branch_enabled:
                retrieval_feat = feat if self.neck_feat == 'after' else weighted_global_feat
                if return_feature_dict:
                    return {
                        "retrieval_feat": retrieval_feat,
                        "global_feat": weighted_global_feat,
                        "global_bn_feat": feat,
                        "local_feats": local_group_feats,
                        "local_bn_feats": local_group_bn_feats,
                        "patch_weights": patch_weights.squeeze(-1) if patch_weights is not None else None,
                        "patch_tokens": patch_tokens,
                        "jpm_local_feats": [local_feat_1, local_feat_2, local_feat_3, local_feat_4],
                        "jpm_local_bn_feats": [local_feat_1_bn, local_feat_2_bn, local_feat_3_bn, local_feat_4_bn],
                    }
                return retrieval_feat

            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [weighted_global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
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
