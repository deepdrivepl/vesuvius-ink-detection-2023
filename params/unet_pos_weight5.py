import numpy as np
from pathlib import Path
from monai.networks.nets import UNet
from torch.nn import BCEWithLogitsLoss
from monai.losses import MaskedLoss
import torch.optim

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
)


PARAMS = dict(
    epochs=1000,
    exp_name="unet_pos_weight5",
    num_z_slices=65,
    batch_size=1,
    acc_batch_size=4,
    patch_size=(512, 512),
    num_samples=12,
    optimizer=torch.optim.RAdam,
    optimizer_params=dict(lr=1e-3, weight_decay=0),
    scheduler=torch.optim.lr_scheduler.OneCycleLR,
    scheduler_params=dict(
        max_lr=3e-4, div_factor=10, final_div_factor=100, pct_start=0.1
    ),
    accelerator="gpu",
    devices=1,
    precision=16,
    num_workers=40,
    seed=2023,
    ckpt_monitor="val/dice",
    # validation
    sw_batch_size=4,
    val_fragment_id="1",
    threshold=0.5,
    # data
    train_data_csv_path=Path("/data") / "data.csv",
    test_data_csv_path=Path("/data") / "test.csv",
)


model_params = dict(
    spatial_dims=2,
    in_channels=PARAMS["num_z_slices"],
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0,
)

model = UNet(**model_params)


def loss_function(outputs, labels, masks):
    loss = BCEWithLogitsLoss(
        pos_weight=torch.tensor([5], dtype=outputs.dtype).to(outputs.device)
    )
    masked_loss = MaskedLoss(loss)(outputs, labels, masks)

    return masked_loss


def get_train_transforms():
    return Compose(
        [
            NormalizeIntensityd(keys="volume_npy", nonzero=True, channel_wise=True),
            # ScaleIntensityd(keys="volume_npy"),
            RandCropByPosNegLabeld(
                keys=("volume_npy", "mask_npy", "label_npy"),
                label_key="label_npy",
                spatial_size=PARAMS["patch_size"],
                num_samples=PARAMS["num_samples"],
                image_key="volume_npy",
                image_threshold=0,
            ),
            RandAffined(
                keys=("volume_npy", "mask_npy", "label_npy"),
                prob=0.75,
                rotate_range=(np.pi / 4, np.pi / 4),
                translate_range=(0.0625, 0.0625),
                scale_range=(0.1, 0.1),
            ),
            RandFlipd(
                keys=("volume_npy", "mask_npy", "label_npy"), spatial_axis=0, prob=0.5
            ),
            RandFlipd(
                keys=("volume_npy", "mask_npy", "label_npy"), spatial_axis=1, prob=0.5
            ),
            RandGaussianNoised(keys="volume_npy", prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys="volume_npy", prob=0.15, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15)
            ),
            RandScaleIntensityd(keys="volume_npy", factors=0.3, prob=0.15),
            RandZoomd(
                keys=("volume_npy", "mask_npy", "label_npy"),
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("bilinear", "nearest", "nearest"),
                align_corners=(True, None, None),
                prob=0.15,
            ),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            NormalizeIntensityd(keys="volume_npy", nonzero=True, channel_wise=True),
            # ScaleIntensityd(keys="volume_npy"),
        ]
    )
