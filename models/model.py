import numpy as np
from timesformer.models.vit import TimeSformer
import torch


NUM_CLASSES = 2
EMBED_DIM = 768


class OCTVIT(TimeSformer):

    def __init__(self, img_size=224, patch_size=16, num_classes=2, num_frames=8, attention_type='space',
                 pretrained_model='', finetune=True, finetune_percent=1, **kwargs):
        super(OCTVIT, self).__init__(img_size, patch_size, num_classes, num_frames, attention_type, pretrained_model, **kwargs)
        self.model.num_classes = num_classes

        if finetune:
            for param in self.model.parameters():
                param.requires_grad = False
            if finetune_percent < 1:
                final_blocks = int(len(self.model.blocks) * finetune_percent)
                for blk in self.model.blocks[final_blocks:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                for param in self.model.norm.parameters():
                    param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

        self.model.heads = [
            self.model.head
            # torch.nn.Linear(EMBED_DIM, 3),  # AMD STAGE
            # torch.nn.Linear(EMBED_DIM, 1),   # DRUSEN
            # torch.nn.Linear(EMBED_DIM, 1),   # SDD
            # torch.nn.Linear(EMBED_DIM, 1),   # PED
            # torch.nn.Linear(EMBED_DIM, 1),   # SCAR
            # torch.nn.Linear(EMBED_DIM, 1),   # SRHEM
            # torch.nn.Linear(EMBED_DIM, 1),   # IRF
            # torch.nn.Linear(EMBED_DIM, 1),   # SRF
            # torch.nn.Linear(EMBED_DIM, 1),   # IRORA
            # torch.nn.Linear(EMBED_DIM, 13),  # IRORA location
            # torch.nn.Linear(EMBED_DIM, 1),   # CRORA
            # torch.nn.Linear(EMBED_DIM, 13),  # cRORA location
        ]
        self.model.heads = torch.nn.ModuleList(self.model.heads)


class VITSmallLinear(OCTVIT):

    def __init__(self, img_size=224, patch_size=16, num_classes=2, num_frames=8,
                 attention_type='divided_space_time',  pretrained_model='', finetune=True,
                 finetune_percent=1, **kwargs):
        super(VITSmallLinear, self).__init__(img_size, patch_size, num_classes, num_frames, attention_type,
                                             pretrained_model, finetune, finetune_percent, **kwargs)

    def forward(self, x):
        # print("\tIn Model: input size", x.size())
        x = self.model.forward_features(x)
        x = [self.model.heads[i](x) for i in range(len(self.model.heads))]
        # print("\toutput size", x[0].size())
        return x


class VITHRLinear(OCTVIT):

    def __init__(self, img_size=448, patch_size=16, num_classes=2, num_frames=16,
                 attention_type='divided_space_time',  pretrained_model='', finetune=True,
                 finetune_percent=1, **kwargs):
        super(VITHRLinear, self).__init__(img_size, patch_size, num_classes, num_frames, attention_type,
                                          pretrained_model, finetune, finetune_percent, **kwargs)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = [self.model.heads[i](x) for i in range(len(self.model.heads))]
        return x


class TIMESFORMERL(OCTVIT):

    def __init__(self, img_size=224, patch_size=16, num_classes=2, num_frames=96,
                 attention_type='space_only',  pretrained_model='', finetune=True,
                 finetune_percent=1, **kwargs):
        super(TIMESFORMERL, self).__init__(img_size, patch_size, num_classes, num_frames, attention_type,
                                          pretrained_model, finetune, finetune_percent, **kwargs)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = [self.model.heads[i](x) for i in range(len(self.model.heads))]
        return x


class VITSmallAvg(OCTVIT):
    def __init__(self, img_size=448, patch_size=16, num_classes=2, num_frames=16,
                 attention_type='divided_space_time',  pretrained_model='', finetune=True,
                 finetune_percent=1, **kwargs):
        super(VITSmallAvg, self).__init__(img_size, patch_size, num_classes, num_frames, attention_type,
                                          pretrained_model, finetune, finetune_percent, **kwargs)

    def forward(self, x):
        f = np.zeros((len(x), self.model.num_classes))
        for i,slice in enumerate(x):
            f[i] = self.model.forward_features(slice)

        x = np.mean(f, axis=1)
        x = self.model.forward_features(x)
        x = [self.model.heads[i](x) for i in range(12)]
        return x


def create_model(cfg):
    # Create a model according to the config file.
    if cfg.MODEL.NUM_CLASSES < 3:
        num_classes = 1
    else:
        num_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.MODEL_NAME == "VIT_8x224_Linear":
        model = VITSmallLinear(
            img_size=cfg.DATA.IMG_SIZE,
            num_classes=num_classes,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            pretrained_model=cfg.MODEL.PRETRAINED_PATH,
            finetune=cfg.MODEL.FINETUNE,
            finetune_percent=cfg.MODEL.FINETUNE_PERCENT
        )
    elif cfg.MODEL.MODEL_NAME == "VIT-HR_16x448_Linear":
        model = VITHRLinear(
            img_size=cfg.DATA.IMG_SIZE,
            num_classes=num_classes,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            pretrained_model=cfg.MODEL.PRETRAINED_PATH,
            finetune=cfg.MODEL.FINETUNE,
            finetune_percent=cfg.MODEL.FINETUNE_PERCENT
        )
    elif cfg.MODEL.MODEL_NAME == "VIT_8x224_Linear_avg":
        model = VITSmallAvg(
            img_size=cfg.DATA.IMG_SIZE,
            num_classes=num_classes,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            pretrained_model=cfg.MODEL.PRETRAINED_PATH,
            finetune=cfg.MODEL.FINETUNE,
            finetune_percent=cfg.MODEL.FINETUNE_PERCENT
        )
    elif cfg.MODEL.MODEL_NAME == "TIMESFORMER_L":
        model = TIMESFORMERL(
            img_size=cfg.DATA.IMG_SIZE,
            num_classes=num_classes,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            pretrained_model=cfg.MODEL.PRETRAINED_PATH,
            finetune=cfg.MODEL.FINETUNE,
            finetune_percent=cfg.MODEL.FINETUNE_PERCENT
        )
    else:
        raise NotImplementedError(
            f"Does not support {cfg.MODEL.MODEL_NAME} type of model")

    if cfg.MODEL.FINETUNE_PERCENT == 1:
        optim_params = model.model.head.parameters()
    else:
        optim_params = model.parameters()
    return model, optim_params
