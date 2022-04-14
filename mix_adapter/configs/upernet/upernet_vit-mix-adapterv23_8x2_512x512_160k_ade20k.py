_base_ = 'upernet_vit-mix-adapterv1_8x2_512x512_160k_ade20k.py'

model = dict(backbone=dict(type='ViTMixAdapterv23'))
data = dict(
    # By default, models are trained on 4 GPUs with 4 images per GPU
    samples_per_gpu=4)
