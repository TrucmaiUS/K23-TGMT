# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=feat_dim, use_gpu=torch.cuda.is_available()
    )

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = None

    def _id_loss(score, target):
        if isinstance(score, list):
            if xent is not None:
                aux_losses = [xent(scor, target) for scor in score[1:]]
                main_loss = xent(score[0], target)
            else:
                aux_losses = [F.cross_entropy(scor, target) for scor in score[1:]]
                main_loss = F.cross_entropy(score[0], target)
            if aux_losses:
                return 0.5 * main_loss + 0.5 * sum(aux_losses) / len(aux_losses)
            return main_loss
        if xent is not None:
            return xent(score, target)
        return F.cross_entropy(score, target)

    def _triplet_loss(feat, target):
        if isinstance(feat, list):
            aux_losses = [triplet(feats, target)[0] for feats in feat[1:]]
            main_loss = triplet(feat[0], target)[0]
            if aux_losses:
                return 0.5 * main_loss + 0.5 * sum(aux_losses) / len(aux_losses)
            return main_loss
        return triplet(feat, target)[0]

    def _visibility_regularization(single_aux_outputs):
        if not single_aux_outputs:
            return 0.0
        vis_weights = single_aux_outputs.get('patch_weights')
        if vis_weights is None:
            vis_weights = single_aux_outputs.get('local_weights')
        if vis_weights is None:
            return 0.0

        reg_type = cfg.MODEL.REL_REG_TYPE if cfg.MODEL.RELIABILITY_PIPELINE else cfg.MODEL.VIS_REG_TYPE
        if reg_type == 'entropy':
            return -torch.sum(vis_weights * torch.log(vis_weights + 1e-12), dim=1).mean()
        return 0.0

    def _reliability_targets(occ_mask, patch_grid):
        patch_target = 1.0 - F.adaptive_avg_pool2d(occ_mask.float(), patch_grid)
        return patch_target.flatten(1).clamp(0.0, 1.0)

    def _reliability_supervision(pair_aux_outputs):
        if cfg.MODEL.REL_VIS_LOSS_WEIGHT <= 0.0 or not pair_aux_outputs:
            return 0.0
        occ_aux = pair_aux_outputs.get('occ_aux')
        occ_mask = pair_aux_outputs.get('occ_mask')
        if not occ_aux or occ_mask is None:
            return 0.0

        reliability_logits = occ_aux.get('reliability_logits')
        patch_reliability = occ_aux.get('patch_reliability')
        patch_grid = occ_aux.get('patch_grid')
        if reliability_logits is None or patch_reliability is None or patch_grid is None:
            return 0.0

        rel_target = 1.0 - _reliability_targets(occ_mask, patch_grid)
        # One-sided supervision: only penalize occluded patches that still get
        # high reliability. Non-occluded patches are left to the ReID losses,
        # consistency term, and aggregation path instead of being forced to 1.
        occluded_weight = rel_target
        if torch.count_nonzero(occluded_weight).item() == 0:
            return patch_reliability.sum() * 0.0
        occlusion_penalty = patch_reliability.clamp(0.0, 1.0)
        return (occluded_weight * occlusion_penalty).sum() / (occluded_weight.sum() + 1e-12)

    def _reliable_patch_consistency(pair_aux_outputs):
        if cfg.MODEL.REL_CONSIST_LOSS_WEIGHT <= 0.0 or not pair_aux_outputs:
            return 0.0
        clean_aux = pair_aux_outputs.get('clean_aux')
        occ_aux = pair_aux_outputs.get('occ_aux')
        occ_mask = pair_aux_outputs.get('occ_mask')
        if not clean_aux or not occ_aux or occ_mask is None:
            return 0.0

        clean_tokens = clean_aux.get('patch_tokens')
        occ_tokens = occ_aux.get('patch_tokens')
        clean_rel = clean_aux.get('patch_reliability')
        occ_rel = occ_aux.get('patch_reliability')
        patch_grid = clean_aux.get('patch_grid')
        if clean_tokens is None or occ_tokens is None or clean_rel is None or occ_rel is None or patch_grid is None:
            return 0.0

        rel_target = _reliability_targets(occ_mask, patch_grid)
        clean_tokens = F.normalize(clean_tokens, dim=-1)
        occ_tokens = F.normalize(occ_tokens, dim=-1)
        if cfg.MODEL.REL_DETACH_CONSIST_WEIGHTS:
            clean_rel = clean_rel.detach()
            occ_rel = occ_rel.detach()
        reliable_weights = torch.min(clean_rel, occ_rel) * rel_target
        patch_cosine = 1.0 - (clean_tokens * occ_tokens).sum(dim=-1)
        return (reliable_weights * patch_cosine).sum() / (reliable_weights.sum() + 1e-12)

    def _reliability_metric_loss(target, pair_aux_outputs):
        if cfg.MODEL.REL_METRIC_LOSS_WEIGHT <= 0.0 or not pair_aux_outputs:
            return 0.0
        clean_aux = pair_aux_outputs.get('clean_aux')
        occ_aux = pair_aux_outputs.get('occ_aux')
        if not clean_aux or not occ_aux:
            return 0.0

        clean_feat = clean_aux.get('reliability_feat')
        occ_feat = occ_aux.get('reliability_feat')
        if clean_feat is None or occ_feat is None:
            return 0.0

        return 0.5 * triplet(clean_feat, target)[0] + 0.5 * triplet(occ_feat, target)[0]

    def _pair_regularization(pair_aux_outputs):
        if not pair_aux_outputs:
            return 0.0
        clean_aux = pair_aux_outputs.get('clean_aux')
        occ_aux = pair_aux_outputs.get('occ_aux')
        reg_terms = []
        if clean_aux:
            reg_terms.append(_visibility_regularization(clean_aux))
        if occ_aux:
            reg_terms.append(_visibility_regularization(occ_aux))
        if not reg_terms:
            return 0.0
        return sum(reg_terms) / len(reg_terms)

    def _base_loss(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE != 'triplet':
            print('expected METRIC_LOSS_TYPE should be triplet'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
            return 0.0

        id_loss = _id_loss(score, target)
        tri_loss = _triplet_loss(feat, target)
        return cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam=None, aux_outputs=None):
            if cfg.MODEL.RELIABILITY_PIPELINE and aux_outputs and aux_outputs.get('occ_score') is not None:
                clean_loss = _id_loss(score, target)
                occ_loss = _id_loss(aux_outputs['occ_score'], target)
                reg_weight = cfg.MODEL.REL_REG_WEIGHT
                total_loss = 0.5 * (clean_loss + occ_loss)
                total_loss = total_loss + reg_weight * _pair_regularization(aux_outputs)
                total_loss = total_loss + cfg.MODEL.REL_VIS_LOSS_WEIGHT * _reliability_supervision(aux_outputs)
                total_loss = total_loss + cfg.MODEL.REL_CONSIST_LOSS_WEIGHT * _reliable_patch_consistency(aux_outputs)
                return total_loss
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, aux_outputs=None):
            if cfg.MODEL.RELIABILITY_PIPELINE and aux_outputs and aux_outputs.get('occ_score') is not None:
                occ_score = aux_outputs['occ_score']
                occ_feat = aux_outputs['occ_feat']
                clean_loss = _base_loss(score, feat, target)
                occ_loss = _base_loss(occ_score, occ_feat, target)
                total_loss = 0.5 * (clean_loss + occ_loss)
                total_loss = total_loss + cfg.MODEL.REL_REG_WEIGHT * _pair_regularization(aux_outputs)
                total_loss = total_loss + cfg.MODEL.REL_VIS_LOSS_WEIGHT * _reliability_supervision(aux_outputs)
                total_loss = total_loss + cfg.MODEL.REL_CONSIST_LOSS_WEIGHT * _reliable_patch_consistency(aux_outputs)
                total_loss = total_loss + cfg.MODEL.REL_METRIC_LOSS_WEIGHT * _reliability_metric_loss(target, aux_outputs)
                return total_loss

            base_loss = _base_loss(score, feat, target)
            return base_loss + cfg.MODEL.VIS_REG_WEIGHT * _visibility_regularization(aux_outputs)

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion
