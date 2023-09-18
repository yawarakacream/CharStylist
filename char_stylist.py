import copy
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "stable_diffusion"))

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import ConcatDataset, DataLoader

from tqdm import tqdm

from diffusers import AutoencoderKL

from layer import UNetModel
from model import EMA, Diffusion
from utility import char2code, save_images


STABLE_DIFFUSION_CHANNEL = 4


class CharStylist:
    def __init__(
        self,
        
        save_path,
        stable_diffusion_path,
        
        char2idx,
        writer2idx,
        
        image_size,
        dim_char_embedding,
        num_res_blocks,
        num_heads,
        
        learning_rate,
        ema_beta,
        diffusion_noise_steps,
        diffusion_beta_start,
        diffusion_beta_end,
        
        diversity_lambda,
        
        device,
    ):
        self.device = device
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        
        self.char2idx = char2idx
        with open(os.path.join(self.save_path, "char2idx.json"), "w") as f:
            json.dump(self.char2idx, f)
        
        self.writer2idx = writer2idx
        with open(os.path.join(self.save_path, "writer2idx.json"), "w") as f:
            json.dump(self.writer2idx, f)
        
        # model info
        
        self.image_size = image_size
        self.dim_char_embedding = dim_char_embedding
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        
        self.learning_rate = learning_rate
        self.ema_beta = ema_beta
        self.diffusion_noise_steps = diffusion_noise_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        
        self.diversity_lambda = diversity_lambda
        
        with open(os.path.join(self.save_path, "model_info.json"), "w") as f:
            info = {
                "image_size": self.image_size,
                "dim_char_embedding": self.dim_char_embedding,
                "num_res_blocks": self.num_res_blocks,
                "num_heads": self.num_heads,
                
                "learning_rate": self.learning_rate,
                "ema_beta": self.ema_beta,
                "diffusion_noise_steps": self.diffusion_noise_steps,
                "diffusion_beta_start": self.diffusion_beta_start,
                "diffusion_beta_end": self.diffusion_beta_end,
                
                "diversity_lambda": self.diversity_lambda,
            }
            json.dump(info, f)
        
        # create layers
        
        self.unet = UNetModel(
            image_size=self.image_size,
            in_channels=STABLE_DIFFUSION_CHANNEL,
            model_channels=self.dim_char_embedding,
            out_channels=STABLE_DIFFUSION_CHANNEL,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_classes=len(self.writer2idx),
            num_heads=self.num_heads,
            context_dim=self.dim_char_embedding,
            vocab_size=len(self.char2idx),
        ).to(device)
        
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        
        self.ema = EMA(self.ema_beta)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        
        self.diffusion = Diffusion(
            noise_steps=self.diffusion_noise_steps,
            beta_start=self.diffusion_beta_start,
            beta_end=self.diffusion_beta_end,
            image_size=self.image_size,
            device=self.device,
        )
    
    @staticmethod
    def load(
        save_path,
        stable_diffusion_path,
        
        device,
    ):
        print("loading CharStylist...")
        
        if not os.path.exists(save_path):
            raise Exception(f"not found: {save_path}")
            
        with open(os.path.join(save_path, "char2idx.json")) as f:
            char2idx = json.load(f)
        
        with open(os.path.join(save_path, "writer2idx.json")) as f:
            writer2idx = json.load(f)
        
        with open(os.path.join(save_path, "model_info.json")) as f:
            model_info = json.load(f)
            
            image_size = model_info["image_size"]
            dim_char_embedding = model_info["dim_char_embedding"]
            num_res_blocks = model_info["num_res_blocks"]
            num_heads = model_info["num_heads"]
            
            learning_rate = model_info["learning_rate"]
            ema_beta = model_info["ema_beta"]
            diffusion_noise_steps = model_info["diffusion_noise_steps"]
            diffusion_beta_start = model_info["diffusion_beta_start"]
            diffusion_beta_end = model_info["diffusion_beta_end"]
            
            diversity_lambda = model_info["diversity_lambda"]
        
        instance = CharStylist(
            save_path,
            stable_diffusion_path,

            char2idx,
            writer2idx,

            image_size,
            dim_char_embedding,
            num_res_blocks,
            num_heads,

            learning_rate,
            ema_beta,
            diffusion_noise_steps,
            diffusion_beta_start,
            diffusion_beta_end,
            
            diversity_lambda,

            device,
        )
        
        print("\tloading UNetModel...")
        
        instance.unet.load_state_dict(torch.load(os.path.join(save_path, "models", "unet.pt")))
        instance.unet.eval() # これいる？
        
        print("\tloaded.")
        
        print("\tloading optimizer...")
        
        instance.optimizer.load_state_dict(torch.load(os.path.join(save_path, "models", "optimizer.pt")))
        
        print("\tloaded.")
        
        print("\tloading EMA...")
        
        instance.ema_model.load_state_dict(torch.load(os.path.join(save_path, "models", "ema_model.pt")))
        instance.ema_model.eval() # これいる？
        
        print("\tloaded.")
        
        print("loaded.")
        
        return instance
    
    def train(self, data_loader: DataLoader[ConcatDataset], epochs, test_chars, test_writers):
        if os.path.exists(os.path.join(self.save_path, "train_info.json")):
            raise Exception("already trained")
        
        os.makedirs(os.path.join(self.save_path, "models"))
        os.makedirs(os.path.join(self.save_path, "generated"))
        
        num_epochs_digit = len(str(epochs))
        
        if epochs < 1000:
            checkpoint_epochs = set([0, epochs - 1])
        else:
            # ex) epochs = 1000 => checkpoint_epochs = {0, 99, 199, ..., 899, 999}
            tmp = 10 ** (num_epochs_digit - 2)
            checkpoint_epochs = set(i - 1 for i in range(tmp, epochs, tmp))
            checkpoint_epochs.add(0)
            checkpoint_epochs.add(epochs - 1)
            del tmp
        
        mse_loss = nn.MSELoss()
        
        loss_list = []

        self.unet.train()
        
        for epoch in range(epochs):
            loss_sum = 0

            pbar = tqdm(data_loader, desc=f"{epoch=}")
            for i, (images, chars_idx, writers_idx) in enumerate(pbar):
                images = images.to(self.device)
                original_images = images
                
                images = self.vae.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
                
                chars_idx = chars_idx.to(self.device)
                writers_idx = writers_idx.to(self.device)

                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)

                if np.random.random() < 0.1:
                    labels = None

                predicted_noise = self.unet(
                    x_t, original_images=original_images, timesteps=t, context=chars_idx, y=writers_idx, or_images=None
                )

                loss = mse_loss(noise, predicted_noise)
                
                # L_div
                perm0 = np.random.permutation(images.shape[0])
                perm1 = np.random.permutation(images.shape[0])
                for i, j in zip(perm0, perm1):
                    if i == j:
                        continue
                    loss -= self.diversity_lambda * nn.functional.l1_loss(predicted_noise[i], predicted_noise[j]) / nn.functional.l1_loss(images[i], images[j])
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.unet)
                
                pbar.set_postfix(MSE=loss.item())

                loss_sum += loss.item()

            loss_list.append(loss_sum / len(pbar))

            if epoch in checkpoint_epochs:
                for test_char in test_chars:
                    images = self.sampling(test_char, test_writers)
                    path = os.path.join(self.save_path, "generated", f"test_{char2code(test_char)}_{str(epoch + 1).zfill(num_epochs_digit)}.jpg")
                    save_images(images, path)

                torch.save(self.unet.state_dict(), os.path.join(self.save_path, "models", "unet.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "models", "optimizer.pt"))
                torch.save(self.ema_model.state_dict(), os.path.join(self.save_path, "models", "ema_model.pt"))
        
        with open(os.path.join(self.save_path, "train_info.json"), "w") as f:
            info = {
                "data_loader": {
                    "batch_size": data_loader.batch_size,
                    "datasets": [d.cs_dataset_info for d in data_loader.dataset.datasets],
                },
                "epochs": epochs,
                "loss": loss_list,
                "test": {
                    "chars": test_chars,
                    "writers": test_writers,
                },
            }
            json.dump(info, f)
    
    def sampling(self, char, writers):
        char_idx = self.char2idx[char]
        writers_idx = [self.writer2idx[w] for w in writers]
        
        ema_sampled_images = self.diffusion.sampling(
            self.ema_model, self.vae, char_idx, writers_idx
        )
        
        return ema_sampled_images
