# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .patch_weight_regularization import PatchWeightRegularization


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=torch.cuda.is_available()
    )  # center loss
    patch_weight_regularizer = PatchWeightRegularization(
        target_mean=cfg.MODEL.VIS_WEIGHT.TARGET_MEAN,
        alpha_reg=cfg.MODEL.VIS_WEIGHT.ALPHA_REG,
        beta_reg=cfg.MODEL.VIS_WEIGHT.BETA_REG,
    )
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def id_loss(logits, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            return xent(logits, target)
        return F.cross_entropy(logits, target)

    def semantic_alignment_losses(outputs):
        semantic_tokens = outputs.get("semantic_tokens")
        semantic_reference_tokens = outputs.get("semantic_reference_tokens")
        visible_mask = outputs.get("semantic_visible_mask")
        if semantic_tokens is None or semantic_reference_tokens is None or visible_mask is None:
            zero = outputs["global_feat"].new_tensor(0.0)
            return zero, zero

        semantic_tokens = F.normalize(semantic_tokens, dim=-1)
        semantic_reference_tokens = F.normalize(semantic_reference_tokens, dim=-1)
        visible_mask = visible_mask.float()

        positive_distance = 1.0 - (semantic_tokens * semantic_reference_tokens).sum(dim=-1)
        align_loss = (positive_distance * visible_mask).sum() / (visible_mask.sum() + cfg.MODEL.SEM_ALIGN.EPS)

        temperature = max(cfg.MODEL.SEM_ALIGN.MATCH_TEMPERATURE, cfg.MODEL.SEM_ALIGN.EPS)
        similarity_matrix = torch.matmul(semantic_tokens, semantic_reference_tokens.transpose(1, 2)) / temperature
        invalid_part_mask = (visible_mask < 0.5).unsqueeze(1)
        targets = torch.arange(similarity_matrix.size(-1), device=similarity_matrix.device)
        targets = targets.unsqueeze(0).expand(similarity_matrix.size(0), -1)

        row_logits = similarity_matrix.masked_fill(invalid_part_mask, -1e4)
        row_log_prob = F.log_softmax(row_logits, dim=-1)
        row_loss = -row_log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        row_loss = (row_loss * visible_mask).sum() / (visible_mask.sum() + cfg.MODEL.SEM_ALIGN.EPS)

        col_logits = similarity_matrix.transpose(1, 2).masked_fill(invalid_part_mask, -1e4)
        col_log_prob = F.log_softmax(col_logits, dim=-1)
        col_loss = -col_log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        col_loss = (col_loss * visible_mask).sum() / (visible_mask.sum() + cfg.MODEL.SEM_ALIGN.EPS)

        part_match_loss = 0.5 * (row_loss + col_loss)
        return align_loss, part_match_loss

    def semantic_batch_prototype_loss(outputs, target):
        semantic_tokens = outputs.get("semantic_tokens")
        visible_mask = outputs.get("semantic_visible_mask")
        if semantic_tokens is None or visible_mask is None:
            return outputs["global_feat"].new_tensor(0.0)

        semantic_tokens = F.normalize(semantic_tokens, dim=-1)
        prototype_tokens = semantic_tokens.detach() if cfg.MODEL.SEM_ALIGN.DETACH_BATCH_PROTOTYPES else semantic_tokens
        visible_mask = visible_mask > 0.5
        temperature = max(cfg.MODEL.SEM_ALIGN.MATCH_TEMPERATURE, cfg.MODEL.SEM_ALIGN.EPS)
        losses = []

        for part_index in range(semantic_tokens.size(1)):
            part_visible = visible_mask[:, part_index]
            visible_indices = torch.nonzero(part_visible, as_tuple=False).squeeze(1)
            if visible_indices.numel() < 2:
                continue

            part_anchor_tokens = semantic_tokens[visible_indices, part_index]
            part_proto_tokens = prototype_tokens[visible_indices, part_index]
            part_targets = target[visible_indices]
            unique_ids, inverse = torch.unique(part_targets, sorted=False, return_inverse=True)

            if unique_ids.numel() < 2:
                continue

            prototype_sums = part_proto_tokens.new_zeros(unique_ids.numel(), part_proto_tokens.size(-1))
            prototype_sums.index_add_(0, inverse, part_proto_tokens)
            prototype_counts = torch.bincount(inverse, minlength=unique_ids.numel()).to(part_proto_tokens.dtype)

            for anchor_index in range(part_anchor_tokens.size(0)):
                positive_group = inverse[anchor_index].item()

                candidate_sums = prototype_sums.clone()
                candidate_counts = prototype_counts.clone()
                candidate_sums[positive_group] -= part_proto_tokens[anchor_index]
                candidate_counts[positive_group] -= 1

                valid_groups = candidate_counts > 0
                if valid_groups.sum().item() < 2 or not valid_groups[positive_group]:
                    continue

                group_indices = torch.nonzero(valid_groups, as_tuple=False).squeeze(1)
                positive_index = torch.nonzero(group_indices == positive_group, as_tuple=False).view(-1).item()
                candidate_prototypes = candidate_sums[group_indices] / candidate_counts[group_indices].unsqueeze(1)
                candidate_prototypes = F.normalize(candidate_prototypes, dim=-1)

                logits = torch.matmul(
                    part_anchor_tokens[anchor_index:anchor_index + 1],
                    candidate_prototypes.transpose(0, 1)
                ) / temperature
                losses.append(F.cross_entropy(logits, torch.tensor([positive_index], device=logits.device)))

        if not losses:
            return outputs["global_feat"].new_tensor(0.0)
        return torch.stack(losses).mean()

    def semantic_patch_supervision_loss(outputs):
        semantic_patch_logits = outputs.get("semantic_patch_logits")
        semantic_patch_targets = outputs.get("semantic_patch_targets")
        if semantic_patch_logits is None or semantic_patch_targets is None:
            return outputs["global_feat"].new_tensor(0.0)

        patch_log_prob = F.log_softmax(semantic_patch_logits, dim=-1)
        patch_loss = -(semantic_patch_targets * patch_log_prob).sum(dim=-1)
        foreground_mass = 1.0 - semantic_patch_targets[..., 0]
        patch_weights = 0.25 + 0.75 * foreground_mass
        return (patch_loss * patch_weights).sum() / (patch_weights.sum() + cfg.MODEL.SEM_ALIGN.EPS)

    def semantic_pixel_decoder_loss(outputs):
        semantic_pixel_logits = outputs.get("semantic_pixel_logits")
        semantic_pixel_targets = outputs.get("semantic_pixel_targets")
        if semantic_pixel_logits is None or semantic_pixel_targets is None:
            return outputs["global_feat"].new_tensor(0.0)

        semantic_pixel_targets = semantic_pixel_targets.long().clamp(0, cfg.MODEL.SEM_ALIGN.NUM_PARTS)
        if semantic_pixel_logits.shape[-2:] != semantic_pixel_targets.shape[-2:]:
            semantic_pixel_logits = F.interpolate(
                semantic_pixel_logits,
                size=semantic_pixel_targets.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        pixel_loss = F.cross_entropy(semantic_pixel_logits, semantic_pixel_targets, reduction='none')
        bg_weight = float(cfg.MODEL.SEM_ALIGN.PIXEL_LOSS_BG_WEIGHT)
        pixel_weights = torch.where(
            semantic_pixel_targets > 0,
            torch.ones_like(pixel_loss),
            pixel_loss.new_full(pixel_loss.shape, bg_weight),
        )
        return (pixel_loss * pixel_weights).sum() / (pixel_weights.sum() + cfg.MODEL.SEM_ALIGN.EPS)

    def legacy_softmax_triplet_loss(score, feat, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            if isinstance(score, list):
                id_losses = [xent(scor, target) for scor in score[1:]]
                id_losses = sum(id_losses) / len(id_losses)
                id_losses = 0.5 * id_losses + 0.5 * xent(score[0], target)
            else:
                id_losses = xent(score, target)
        else:
            if isinstance(score, list):
                id_losses = [F.cross_entropy(scor, target) for scor in score[1:]]
                id_losses = sum(id_losses) / len(id_losses)
                id_losses = 0.5 * id_losses + 0.5 * F.cross_entropy(score[0], target)
            else:
                id_losses = F.cross_entropy(score, target)

        if isinstance(feat, list):
            tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
            tri_losses = sum(tri_losses) / len(tri_losses)
            tri_losses = 0.5 * tri_losses + 0.5 * triplet(feat[0], target)[0]
        else:
            tri_losses = triplet(feat, target)[0]

        return cfg.MODEL.ID_LOSS_WEIGHT * id_losses + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_losses

    def transformer_dict_loss(outputs, target):
        if "base_scores" in outputs and "base_feats" in outputs:
            total_loss = legacy_softmax_triplet_loss(outputs["base_scores"], outputs["base_feats"], target)
        else:
            total_loss = cfg.MODEL.ID_LOSS_WEIGHT * id_loss(outputs["global_logits"], target)

            if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
                total_loss = total_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(outputs["global_feat"], target)[0]

        local_logits = outputs.get("local_logits", [])
        if local_logits:
            local_id_loss = sum(id_loss(local_logit, target) for local_logit in local_logits)
            total_loss = total_loss + cfg.MODEL.LOCAL_GROUP.LAMBDA_ID * local_id_loss

        patch_weights = outputs.get("patch_weights")
        if patch_weights is not None:
            total_loss = total_loss + cfg.MODEL.VIS_WEIGHT.LAMBDA_REG * patch_weight_regularizer(patch_weights)

        align_loss, separation_loss = semantic_alignment_losses(outputs)
        batch_proto_loss = semantic_batch_prototype_loss(outputs, target)
        patch_semantic_loss = semantic_patch_supervision_loss(outputs)
        pixel_semantic_loss = semantic_pixel_decoder_loss(outputs)
        semantic_loss_scale = outputs.get("semantic_loss_scale", 1.0)
        total_loss = total_loss + semantic_loss_scale * cfg.MODEL.SEM_ALIGN.LAMBDA_ALIGN * align_loss
        total_loss = total_loss + semantic_loss_scale * cfg.MODEL.SEM_ALIGN.LAMBDA_SEP * separation_loss
        total_loss = total_loss + semantic_loss_scale * cfg.MODEL.SEM_ALIGN.LAMBDA_BATCH * batch_proto_loss
        total_loss = total_loss + semantic_loss_scale * cfg.MODEL.SEM_ALIGN.LAMBDA_PATCH * patch_semantic_loss
        total_loss = total_loss + semantic_loss_scale * cfg.MODEL.SEM_ALIGN.LAMBDA_PIXEL * pixel_semantic_loss

        return total_loss

    if sampler == 'softmax':
        def loss_func(outputs, target, target_cam=None):
            if isinstance(outputs, dict):
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss(outputs["global_logits"], target)
            score, _ = outputs
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(outputs, target, target_cam=None):
            if isinstance(outputs, dict):
                return transformer_dict_loss(outputs, target)

            score, feat = outputs
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                return legacy_softmax_triplet_loss(score, feat, target)
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


