_base_ = 'upernet_vit-mix-adapterv1_8x2_512x512_160k_ade20k.py'

model = dict(backbone=dict(type='ViTMixAdapterv6'))
