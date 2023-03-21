import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

class InstanceBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, inst_dim, inst_convs, kernel_dim):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
        dim = inst_dim
        num_convs = inst_convs
        num_masks = out_channels
        kernel_dim = kernel_dim
        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        # self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        # self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # for module in [self.iam_conv, self.cls_score]:
        #     nn.init.constant_(module.bias, bias_value)
        nn.init.normal_(self.iam_conv.weight, std=0.01)
        # nn.init.normal_(self.cls_score.weight, std=0.01)

        nn.init.normal_(self.mask_kernel.weight, std=0.01)
        nn.init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        # pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        # pred_scores = self.objectness(inst_features)
        # return pred_logits, pred_kernel, pred_scores, iam
        return pred_kernel

class MaskBranch(nn.Module):

    def __init__(self,in_channels, mask_dim, mask_convs, kernel_dim):
        super().__init__()
        dim = mask_dim
        num_convs = mask_convs
        kernel_dim = kernel_dim
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)

class BaseIAMDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,
                 mask_dim, mask_convs, kernel_dim,
                 inst_dim, inst_convs, output_iam=False):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels + 2

        self.scale_factor = 1 # cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(in_channels, out_channels, num_classes, inst_dim, inst_convs, kernel_dim)
        self.mask_branch = MaskBranch(in_channels, mask_dim, mask_convs, kernel_dim)
        

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)

        # pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        pred_kernel = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        return pred_masks
        # output = {
        #     "pred_logits": pred_logits,
        #     "pred_masks": pred_masks,
        #     "pred_scores": pred_scores,
        # }

        # if self.output_iam:
        #     iam = F.interpolate(iam, scale_factor=self.scale_factor,
        #                         mode='bilinear', align_corners=False)
        #     output['pred_iam'] = iam

        # return output