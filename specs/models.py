"""Holds specifications for models."""
import options.model_options as mo
from models import i3d, tadn, tarn

########################################################################################################################
# TimeAlignedDenseNet
########################################################################################################################
tadn_class = mo.TADNOptions(
    arch=tadn.TimeAlignedDenseNet,
    time_steps=4,
    temporal_in_planes=256,
    growth_rate=64,
    temporal_drop_rate=0.0,
    classifier_drop_rate=0.5,
    class_embed_planes=512
)
########################################################################################################################
# TimeAlignedResNet
########################################################################################################################
tarn_class = mo.TARNOptions(
    arch=tarn.TimeAlignedResNet,
    time_steps=4,
    classifier_drop_rate=0.5,
    temporal_out_planes=128,
    class_embed_planes=512,
    encoder_planes=(16, 32, 64, 128, 256)
)
tarn_ae_large = mo.AETARNOptions(
    arch=tarn.AETimeAlignedResNet,
    time_steps=4,
    classifier_drop_rate=0.5,
    temporal_out_planes=128,
    class_embed_planes=512,
    encoder_planes=(16, 32, 64, 128, 256),
    decoder_planes=(256, 128, 64, 32, 16)
)
tarn_vae_large = mo.VAETARNOptions(
    arch=tarn.VAETimeAlignedResNet,
    time_steps=4,
    classifier_drop_rate=0.5,
    temporal_out_planes=128,
    class_embed_planes=512,
    encoder_planes=(16, 32, 64, 128, 256),
    decoder_planes=(256, 128, 64, 32, 16),
    vote_type='soft',
)
########################################################################################################################
# I3D
########################################################################################################################
i3d_class = mo.I3DOptions(
    arch=i3d.I3D,
    dropout_prob=0.5,
)
i3d_ae_large = mo.AEI3DOptions(
    arch=i3d.AEI3D,
    embed_planes=1024,
    dropout_prob=0.5,
)
i3d_vae_large = mo.VAEI3DOptions(
    arch=i3d.VAEI3D,
    latent_planes=1024,
    dropout_prob=0.5,
    vote_type='soft'
)
