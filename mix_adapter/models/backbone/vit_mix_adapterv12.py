import math

import torch
from mmcv.cnn import build_norm_layer, build_conv_layer, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_, kaiming_init, constant_init
from mmcv.runner import ModuleList, Sequential, BaseModule, CheckpointLoader, load_state_dict
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models import BACKBONES
from mmseg.models.backbones import VisionTransformer
from mmseg.models.utils import nchw2nlc2nchw, nchw_to_nlc
from mmseg.utils import get_root_logger


class ResBlockPreAct(BaseModule):
    def __init__(self,
                 in_channels=768,
                 channels=768,
                 dropout_layer=None,
                 init_cfg=None):
        super(ResBlockPreAct, self).__init__(init_cfg)
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               3,
                               1,
                               1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               3,
                               1,
                               1)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

        if channels != in_channels:
            self.identity_map = nn.Conv2d(in_channels,
                                          channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchw2nlc2nchw(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchw2nlc2nchw(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.dropout_layer(out) + self.identity_map(x)

        return out


class MixerModulev7(BaseModule):
    def __init__(self,
                 num_channels=768,
                 norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
                 dropout_layer=None,
                 init_cfg=None):
        super(MixerModulev7, self).__init__(init_cfg)
        self.mix_linear = nn.Linear(25, 25)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

        self.vit_norm = build_norm_layer(norm_cfg, num_channels)[1]
        self.f8_norm = build_norm_layer(norm_cfg, num_channels)[1]
        self.f16_norm = build_norm_layer(norm_cfg, num_channels)[1]
        self.f32_norm = build_norm_layer(norm_cfg, num_channels)[1]

        self.f8_norm2 = build_norm_layer(norm_cfg, num_channels)[1]
        self.f16_norm2 = build_norm_layer(norm_cfg, num_channels)[1]
        self.f32_norm2 = build_norm_layer(norm_cfg, num_channels)[1]

        self.ffn = ResBlockPreAct(in_channels=num_channels,
                                  channels=num_channels,
                                  dropout_layer=dropout_layer)

        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.f16_mixer = nn.Conv2d(num_channels * 2,
                                   num_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, vit_feature, feature8, feature16, feature32):
        # Their shapes are B, C, H, W
        vit_feature_i, feature8_i, \
        feature16_i, feature32_i = vit_feature, feature8, feature16, feature32

        vit_feature = nchw2nlc2nchw(self.vit_norm, vit_feature)
        feature8 = nchw2nlc2nchw(self.f8_norm, feature8)
        feature16 = nchw2nlc2nchw(self.f16_norm, feature16)
        feature32 = nchw2nlc2nchw(self.f32_norm, feature32)

        b, c, h, w = feature32.shape
        # B, C, 2H, 2W -> B, C, H, 2, W, 2 -> B, H, W, C, 2, 2
        # -> B, H, W, C, 4
        vit_feature = vit_feature.reshape(b, c, h, 2, w, 2). \
            permute(0, 2, 4, 1, 3, 5).contiguous(). \
            view(b, h, w, c, 4).contiguous()
        # B, C, 4H, 4W -> B, C, H, 4, W, 4 -> B, H, W, C, 4, 4
        # -> B, H, W, C, 16
        feature8 = feature8.reshape(b, c, h, 4, w, 4). \
            permute(0, 2, 4, 1, 3, 5).contiguous(). \
            view(b, h, w, c, 16).contiguous()
        # B, C, 2H, 2W -> B, C, H, 2, W, 2 -> B, H, W, C, 2, 2
        # -> B, H, W, C, 4
        feature16 = feature16.reshape(b, c, h, 2, w, 2). \
            permute(0, 2, 4, 1, 3, 5).contiguous(). \
            view(b, h, w, c, 4).contiguous()
        # B, C, H, W -> B, H, W, C, 1
        feature32 = feature32.permute(0, 2, 3, 1).unsqueeze(-1) \
            .contiguous()
        # B, H, W, C, 25
        aligned_feature = torch.cat((vit_feature,
                                     feature8,
                                     feature16,
                                     feature32),
                                    dim=4)
        aligned_feature = self.mix_linear(aligned_feature)
        # B, H, W, C, 4
        vit_feature = aligned_feature[:, :, :, :, 0: 4]
        # B, H, W, C, 16
        feature8 = aligned_feature[:, :, :, :, 4: 20]
        # B, H, W, C, 4
        feature16 = aligned_feature[:, :, :, :, 20: 24]
        # B, H, W, C
        feature32 = aligned_feature[:, :, :, :, 24]

        # B, H, W, C, 4 -> B, H, W, C, 2, 2 -> B, C, H, 2, W, 2
        # -> B, C, 2H, 2W
        vit_feature = vit_feature.reshape(b, h, w, c, 2, 2). \
            permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, 2 * h, 2 * w). \
            contiguous()
        # B, H, W, C, 16 -> B, H, W, C, 4, 4 -> B, C, H, 4, W, 4
        # -> B, C, 4H, 4W
        feature8 = feature8.reshape(b, h, w, c, 4, 4). \
            permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, 4 * h, 4 * w). \
            contiguous()
        # B, H, W, C, 4 -> B, H, W, C, 2, 2 -> B, C, H, 2, W, 2
        # -> B, C, 2H, 2W
        feature16 = feature16.reshape(b, h, w, c, 2, 2). \
            permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, 2 * h, 2 * w). \
            contiguous()
        # B, H, W, C -> B, C, H, W
        feature32 = feature32.permute(0, 3, 1, 2).contiguous()

        feature8 = self.dropout_layer(feature8) + feature8_i
        feature16 = self.dropout_layer(feature16) + feature16_i
        feature32 = self.dropout_layer(feature32) + feature32_i

        vit_feature = vit_feature_i + self.dropout_layer(self.gamma * vit_feature)
        feature8 = self.ffn(nchw2nlc2nchw(self.f8_norm2,
                                          feature8,
                                          contiguous=True),
                            feature8)
        feature16 = self.ffn(nchw2nlc2nchw(self.f16_norm2,
                                           feature16,
                                           contiguous=True),
                             feature16)
        feature32 = self.ffn(nchw2nlc2nchw(self.f32_norm2,
                                           feature32,
                                           contiguous=True),
                             feature32)
        feature16 = self.f16_mixer(torch.cat((feature16, vit_feature),
                                             dim=1))
        return vit_feature, [feature8, feature16, feature32]


@BACKBONES.register_module()
class ViTMixAdapterv12(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None,

                 stem_channels=64,
                 deep_stem=True,
                 conv_cfg=None,
                 conv_norm_cfg=dict(type='BN', requires_grad=True),
                 num_interactions=4,
                 align_corners=False):
        super(ViTMixAdapterv12, self).__init__(img_size,
                                               patch_size,
                                               in_channels,
                                               embed_dims,
                                               num_layers,
                                               num_heads,
                                               mlp_ratio,
                                               out_indices,
                                               qkv_bias,
                                               drop_rate,
                                               attn_drop_rate,
                                               drop_path_rate,
                                               with_cls_token,
                                               output_cls_token,
                                               norm_cfg,
                                               act_cfg,
                                               patch_norm,
                                               final_norm,
                                               interpolate_mode,
                                               num_fcs,
                                               norm_eval,
                                               with_cp,
                                               pretrained,
                                               init_cfg)
        self.deep_stem = deep_stem
        self.conv_cfg = conv_cfg
        self.stem_channels = stem_channels
        self.conv_norm_cfg = conv_norm_cfg
        self.num_interactions = num_interactions
        self.align_corners = align_corners

        self._make_stem_layer(in_channels, stem_channels)

        self.adap_pyramidizer = ModuleList(Sequential(nn.Conv2d(stem_channels * 2 ** i,
                                                                stem_channels * 2 ** (i + 1),
                                                                kernel_size=3,
                                                                stride=2,
                                                                padding=1,
                                                                bias=False),
                                                      build_norm_layer(self.conv_norm_cfg,
                                                                       stem_channels * 2 ** (i + 1))[1],
                                                      nn.ReLU(inplace=True)) for i in range(3))
        self.adap_c_equalizer = ModuleList(Sequential(nn.Conv2d(stem_channels * 2 ** (i + 1),
                                                                embed_dims,
                                                                1,
                                                                1)) for i in range(3))

        inter_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.num_interactions)
        ]  # stochastic depth decay rule
        self.adap_interactions = ModuleList(MixerModulev7(embed_dims,
                                                          dropout_layer=dict(type='DropPath',
                                                                             drop_prob=inter_dpr[i]))
                                            for i in range(self.num_interactions))
        self.adap_interaction_indexes = [(num_layers // num_interactions) * i - 1
                                         for i in range(1,
                                                        self.num_interactions + 1)]
        self.adap_feature4_generator = nn.Sequential(nn.Upsample(scale_factor=2,
                                                                 mode='bilinear',
                                                                 align_corners=self.align_corners),
                                                     nn.Conv2d(embed_dims,
                                                               embed_dims,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1))

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                    f'{state_dict["pos_embed"].shape} to '
                                    f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=logger)
            for n, m in self.named_children():
                if n.startswith('adap_'):
                    if hasattr(m, 'init_weights'):
                        m.init_weights()
                    else:
                        if isinstance(m, nn.Linear):
                            trunc_normal_(m.weight, std=.02)
                            if m.bias is not None:
                                if 'ffn' in n:
                                    nn.init.normal_(m.bias, mean=0., std=1e-6)
                                else:
                                    nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Conv2d):
                            kaiming_init(m, mode='fan_in', bias=0.)
                        elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                            constant_init(m, val=1.0, bias=0.)

        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.adap_stem = Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.conv_norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.conv_norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.conv_norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.adap_conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.adap_norm1 = build_norm_layer(
                self.conv_norm_cfg, stem_channels, postfix=1)[1]
            self.adap_relu = nn.ReLU(inplace=True)
        self.adap_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.deep_stem:
            adap_x = self.adap_stem(inputs)
        else:
            adap_x = self.adap_conv1(inputs)
            adap_x = self.adap_norm1(adap_x)
            adap_x = self.adap_relu(adap_x)
        adap_x = self.adap_maxpool(adap_x)
        # adap_x : 1/4
        # 1/8, 1/16, 1/32
        pyramid_features = []
        for pyramid_layer in self.adap_pyramidizer:
            adap_x = pyramid_layer(adap_x)
            pyramid_features.append(adap_x)
        # map to same channels
        for i in range(len(pyramid_features)):
            pyramid_features[i] = \
                self.adap_c_equalizer[i](pyramid_features[i])

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.adap_interaction_indexes:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    vit_feature = x[:, 1:]
                else:
                    vit_feature = x
                B, _, C = vit_feature.shape
                vit_feature = vit_feature.reshape(B, hw_shape[0], hw_shape[1],
                                                  C).permute(0, 3, 1, 2).contiguous()
                vit_feature, pyramid_features = self.adap_interactions[
                    self.adap_interaction_indexes.index(i)](vit_feature, *pyramid_features)
                vit_feature = nchw_to_nlc(vit_feature)
                if self.with_cls_token:
                    x = torch.cat((x[:, 0: 1], vit_feature), dim=1)

        pyramid_features.insert(0, self.adap_feature4_generator(pyramid_features[0]))

        return tuple(pyramid_features)
