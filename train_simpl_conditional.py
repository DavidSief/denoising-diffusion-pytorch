from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet_conditional, GaussianDiffusion_conditional, Trainer
import torch


# num classes in Unet
# random lables in GaussianDiffusion.sample()
# dataset select binary_classification 


def train():
    gpu_index = 3
    #device = torch.device(f"cuda:{gpu_index}")

    #try to make Unet conditional and run it with 01 and 234 as classes
    #generate 100% for 01 and 234 respectively
    model = Unet_conditional(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True,
        num_classes=5,
        embed_dim=32,
    )#.to(device)

    diffusion = GaussianDiffusion_conditional(
        model,
        image_size = 224, #224
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )#.to(device)

    trainer = Trainer(
        diffusion,
        '',
        train_batch_size = 60,      #60
        train_lr = 8e-5,
        train_num_steps = 24000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        results_folder = "results/run_no_prost_conditional_5_classes_balanced",      # folder to save results
        save_and_sample_every = 790, #791,      # interval to save and sample results (10 epochs)
        split_batches=True
        )

    trainer.train(conditional_train = True)


if(__name__ == "__main__"):
    train()