ENCODER_CONFIGS = {
    "img_channels": 3,
    "down_channels": (32, 64, 128, 256, 512),
    "expand_factor": 2,
    "drop_p": 0.4,
    "num_groups_norm": 8,
    "activation": "swish",
    "dtype": None,
}

DECODER_CONFIGS = {
    "img_channels": 3,
    "latent_channels": 512,
    "up_channels": (256, 128, 64, 32),
    "expand_factor": 2,
    "drop_p": 0.3,
    "num_groups_norm": 8,
    "activation": "swish",
    "latent_shape": (8, 8),
    "dtype": None,
}

def get_ae_configs(pretrained_encoder: bool = False):
    encoder_configs = {}
    if not pretrained_encoder:
        encoder_configs = ENCODER_CONFIGS
    decoder_configs = DECODER_CONFIGS
    
    return {
        "encoder": encoder_configs,
        "decoder": decoder_configs
    }