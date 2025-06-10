ENCODER_CONFIGS = {
    "img_channels": 3,
    "down_channels": (16, 32, 64),
    "expand_factor": 2,
    "drop_p": 0.35,
    "num_groups_norm": 4,
    "activation": "hardswish",
    "dtype": None,
}

DECODER_CONFIGS = {
    "img_channels": 3,
    "latent_channels": 64,
    "up_channels": (32, 24, 16),
    "expand_factor": 2,
    "drop_p": 0.35,
    "num_groups_norm": 4,
    "activation": "hardswish",
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