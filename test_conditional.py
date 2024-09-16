import torch
import torchvision.utils as vutils
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    Unet_conditional,
    GaussianDiffusion_conditional,
    Trainer,
)
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

from denoising_diffusion_pytorch.datasets import Pingjun


def print_gpu_memory(device):
    """Prints the current and maximum GPU memory allocation."""
    print(
        f"\t-- Current memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2} MB"
    )
    print(
        f"\t-- Current memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2} MB"
    )
    print(
        f"\t-- Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 2} MB"
    )


def main():

    gpu_index = 0
    device = torch.device(f"cuda:{gpu_index}")

    model = Unet_conditional(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
        num_classes=2,
        embed_dim=32,
    ).to(device)

    diffusion = GaussianDiffusion_conditional(
        model,
        image_size=224,  # 224
        timesteps=1000,  # number of steps
        sampling_timesteps=500,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ).to(device)

    output_dir = "generated_images_cond_2_classes/kl01"

    trainer = Trainer(
        diffusion,
        "",
        train_batch_size=60,  # 60
        train_lr=8e-5,
        train_num_steps=24000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        results_folder="results/run_no_prost_conditional_2_classes",  # folder to save results
        save_and_sample_every=790,  # 791,      # interval to save and sample results (10 epochs)
        split_batches=False,
    )

    trainer.load("30")

    image_id = 0

    for _ in range(85):     #85

        with torch.inference_mode():
            batchSize = 200  # [200] * 85 for 17.000 images --> ~9h
            
            labels = [0 for _ in range(batchSize)]
            labels = torch.tensor(labels, device=trainer.device)
            #labels = torch.zeros(batchSize, device=trainer.device)

            all_images_list = trainer.ema.ema_model.sample(batch_size=batchSize, labels = labels)

        
        for j, img_fake in enumerate(all_images_list):

            vutils.save_image(
                img_fake.detach(),
                f"{output_dir}/{image_id}.png",
            )
            ######################################################
            image_id += 1

            print(f"Saved image: {image_id}")
            # print_gpu_memory(device)


#  8.215
# 16.430

if __name__ == "__main__":
    main()
