import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    Simplified U-Net architecture
    """
    def __init__(self, in_channels=4, out_channels=4, time_dim=256, text_dim=768):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Downsampling path
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.down3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        
        # Cross-attention layer
        self.cross_attn = CrossAttention(256, text_dim)
        
        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        
        # Upsampling path
        self.up1 = nn.ConvTranspose2d(512, 128, 4, padding=1, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 64, 4, padding=1, stride=2)
        self.out = nn.Conv2d(128, out_channels, 3, padding=1)
        
    def forward(self, x, t, text_embedding):
        x = x.float()  # or x.to(torch.float32)
        t = t.float()  # This is already done in your code
        text_embedding = text_embedding.float()
        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, time_dim]
        
        # Downsampling
        d1 = F.silu(self.down1(x))
        
        # Add time embedding to d1 (need to reshape t_emb to match d1's dimensions)
        t_emb_d1 = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d1.shape[2], d1.shape[3])
        t_emb_d1 = t_emb_d1[:, :d1.shape[1], :, :]  # Slice to match channels
        d1 = d1 + t_emb_d1
        
        d2 = F.silu(self.down2(d1))
        d3 = F.silu(self.down3(d2))
        
        # Cross-attention with text embeddings
        d3 = self.cross_attn(d3, text_embedding)
        
        # Middle (also condition on time)
        mid = self.mid(d3)
        
        # Upsampling with skip connections
        u1 = torch.cat([mid, d3], dim=1)
        u1 = F.silu(self.up1(u1))
        u2 = torch.cat([u1, d2], dim=1)
        u2 = F.silu(self.up2(u2))
        
        # Output
        out = self.out(torch.cat([u2, d1], dim=1))
        return out


class CrossAttention(nn.Module):
    def __init__(self, channels, text_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        
        self.norm = nn.LayerNorm([channels])
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.to_out = nn.Linear(channels, channels)
        
    def forward(self, x, text_embedding):
        # Reshape spatial dimensions for attention
        x = x.float()
        text_embedding = text_embedding.float()
        
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply layer norm
        x_norm = self.norm(x_flat)
        
        # Project to queries, keys, values
        q = self.to_q(x_norm).reshape(b, h * w, self.heads, c // self.heads).permute(0, 2, 1, 3)  # [B, heads, H*W, C//heads]
        k = self.to_k(text_embedding).reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)  # [B, heads, seq_len, C//heads]
        v = self.to_v(text_embedding).reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)  # [B, heads, seq_len, C//heads]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, h * w, c)
        out = self.to_out(out).permute(0, 2, 1).reshape(b, c, h, w)
        
        # Residual connection
        return x + out


class DiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, dtype=np.float32)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod).astype(np.float32)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod).astype(np.float32)
        self.sqrt_recip_alphas = np.sqrt(1 / self.alphas).astype(np.float32)
        self.posterior_variance = (self.betas[1:] * (1 - self.alphas_cumprod[:-1]) / 
                                (1 - self.alphas_cumprod[1:])).astype(np.float32)
        self.posterior_variance = np.append(self.posterior_variance, 0).astype(np.float32)
        self.posterior_log_variance = np.log(np.maximum(self.posterior_variance, 1e-20)).astype(np.float32)
        self.posterior_mean_coef1 = (self.betas[1:] * np.sqrt(self.alphas_cumprod[:-1]) /
                                    (1 - self.alphas_cumprod[1:])).astype(np.float32)
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod[:-1]) * np.sqrt(self.alphas[1:]) /
                                    (1 - self.alphas_cumprod[1:])).astype(np.float32)
        self.posterior_mean_coef1 = np.append(self.posterior_mean_coef1, 0).astype(np.float32)
        self.posterior_mean_coef2 = np.append(self.posterior_mean_coef2, 0).astype(np.float32)

    
    def q_sample(self, x_0, t):
        """
        Forward diffusion process q(x_t | x_0).
        Apply noise to the initial image according to the diffusion schedule.
        """
        # Convert PyTorch tensor to NumPy if needed
        if isinstance(x_0, torch.Tensor):
            x_0_np = x_0.cpu().numpy()
        else:
            x_0_np = x_0
            
        # Generate random noise
        noise = np.random.randn(*x_0_np.shape)
        
        # Apply noise according to schedule
        mean = self.sqrt_alphas_cumprod[t][:, None, None, None] * x_0_np
        var = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = mean + var * noise
        
        # Convert back to PyTorch tensor if needed
        if isinstance(x_0, torch.Tensor):
            return torch.from_numpy(x_t).to(x_0.device), torch.from_numpy(noise).to(x_0.device)
        return x_t, noise
    
    def p_mean_variance(self, x_t, t, text_embedding):
        """
        Calculate the parameters of the posterior distribution p(x_{t-1} | x_t).
        """
        # Use the model to predict noise
        if not isinstance(x_t, torch.Tensor):
            x_t = torch.from_numpy(x_t).float()
            convert_back = True
        else:
            x_t = x_t.float()
            convert_back = False
        
        text_embedding = text_embedding.float()
            
        t_tensor = torch.tensor(t, dtype=torch.float, device=x_t.device)
        
        # Predict noise with the model
        predicted_noise = self.model(x_t, t_tensor, text_embedding)
        predicted_noise = predicted_noise.float()
        
        # Calculate mean of the posterior
        batch_size = x_t.shape[0]
        posterior_mean = torch.zeros_like(x_t)
        posterior_log_variance = torch.zeros((batch_size, 1, 1, 1), device=x_t.device)
        
        for i in range(batch_size):
            coef1 = torch.tensor(self.posterior_mean_coef1[t[i]], device=x_t.device, dtype=torch.float32)
            coef2 = torch.tensor(self.posterior_mean_coef2[t[i]], device=x_t.device, dtype=torch.float32)

            sqrt_one_minus_alpha_cumprod = torch.tensor(self.sqrt_one_minus_alphas_cumprod[t[i]], device=x_t.device, dtype=torch.float32)
            sqrt_alpha_cumprod = torch.tensor(self.sqrt_alphas_cumprod[t[i]], device=x_t.device, dtype=torch.float32)

            # Calculate predicted x_0
            x_0_pred = (x_t[i] - sqrt_one_minus_alpha_cumprod * predicted_noise[i]) / sqrt_alpha_cumprod

            posterior_mean[i] = coef1 * x_0_pred + coef2 * x_t[i]

            # Convert posterior log variance explicitly
            posterior_log_variance[i] = torch.tensor(self.posterior_log_variance[t[i]], device=x_t.device, dtype=torch.float32)
        
        
        if convert_back:
            posterior_mean = posterior_mean.cpu().numpy()
            posterior_log_variance = posterior_log_variance.cpu().numpy()
            
        return posterior_mean, posterior_log_variance
    
    def p_sample(self, x_t, t, text_embedding):
        """
        Sample from p(x_{t-1} | x_t) using the reparameterization trick.
        """
        mean, log_var = self.p_mean_variance(x_t, t, text_embedding)
        
        # Sample using the reparameterization trick
        if isinstance(mean, np.ndarray):
            noise = np.random.randn(*mean.shape)
            std = np.exp(0.5 * log_var)
            x_t_1 = mean + std * noise
        else:
            noise = torch.randn_like(mean)
            std = torch.exp(0.5 * log_var)
            x_t_1 = mean + std * noise
            
        return x_t_1
    
    def p_sample_loop(self, shape, text_embedding, device="cpu"):
        """
        Generate a sample by iteratively sampling from p(x_{t-1} | x_t).
        """
        text_embedding = text_embedding.float()
        # Start from pure noise
        x_t = torch.randn(shape, device=device, dtype=torch.float32) 
        # x_t = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            print(f"Sampling timestep {t}/{self.timesteps}")
            t_batch = np.full(shape[0], t)
            x_t = self.p_sample(x_t, t_batch, text_embedding)
            
        return x_t
    
    def train_step(self, x_0, text_embedding, optimizer):
        """
        Perform a single training step.
        """
        optimizer.zero_grad()
        x_0 = x_0.float()
        # Sample a random timestep for each image
        batch_size = x_0.shape[0]
        t = np.random.randint(0, self.timesteps, size=batch_size)
        t_tensor = torch.tensor(t, dtype=torch.float, device=x_0.device)
        
        # Forward diffusion process (add noise)
        x_t, noise = self.q_sample(x_0, t)
        
        # Predict the noise using the model
        predicted_noise = self.model(x_t, t_tensor, text_embedding)
        # Predict the noise using the model
        predicted_noise = predicted_noise.float()
        noise_tensor = noise if isinstance(noise, torch.Tensor) else torch.from_numpy(noise).to(x_0.device)
        noise_tensor = noise_tensor.float()
        
        loss = F.mse_loss(predicted_noise, noise_tensor)
        
        # KL divergence component is rarely implemented explicitly in practice
        # This is approximated by the MSE loss above in the diffusion objective
        
        # Backpropagate and update weights
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, max_seq_len=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=12, 
                dim_feedforward=4*embed_dim,
                batch_first=True
            ),
            num_layers=12
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Token + Position embeddings
        input_ids = input_ids.long()
        embeddings = self.token_embedding(input_ids) + self.position_embedding[:, :input_ids.size(1), :]
        
        # Pass through transformer
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create causal mask for transformer
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(input_ids.device)
        
        # Pass through transformer
        output = self.transformer(embeddings, mask=causal_mask, src_key_padding_mask=~attention_mask.bool())
        
        return output


class SubsetDataset(Dataset):
    def __init__(self, root_folder, img_size=64, transform=None):
        self.root_folder = root_folder
        self.images_folder = os.path.join(root_folder, "images")
        self.captions_folder = os.path.join(root_folder, "captions")
        
        # Set up the dataset items
        self.dataset_items = []
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext)))
        
        for image_path in image_files:
            image_filename = os.path.basename(image_path)
            image_id = os.path.splitext(image_filename)[0]  # Remove extension
            
            # Look for a corresponding caption file
            caption_path = os.path.join(self.captions_folder, f"{image_id}.txt")
            
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
                
                self.dataset_items.append({
                    'image_path': image_path,
                    'caption': caption
                })
        
        print(f"Created dataset with {len(self.dataset_items)} matched image-caption pairs")
        
        # Set up transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.dataset_items)
    
    def __getitem__(self, idx):
        item = self.dataset_items[idx]
        
        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': item['caption']
        }

class SimpleTokenizer:
    def __init__(self, vocab_size=50257, max_length=77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def encode(self, text, device="cpu"):
        tokens = []
        for i, char in enumerate(text[:self.max_length-2]):  # Leave room for BOS/EOS
            # Simple hash function to map characters to token IDs
            token_id = (ord(char) * 17) % (self.vocab_size - 3) + 3  # Reserve 0,1,2 for special tokens
            tokens.append(token_id)
            
        # Add special tokens (0=PAD, 1=BOS, 2=EOS)
        tokens = [1] + tokens + [2]
        
        # Pad to max length
        padding = [0] * (self.max_length - len(tokens))
        tokens = tokens + padding
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        attention_mask = attention_mask + [0] * len(padding)
        
        # Convert to tensors
        input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.float, device=device)
        attention_mask = torch.tensor(attention_mask[:self.max_length], dtype=torch.float, device=device)
        
        return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)  # Add batch dimension

def train(
    root_folder="./coco_subset",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    img_size=64,
    timesteps=1000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = SubsetDataset(root_folder, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    unet = SimpleUNet(in_channels=4, out_channels=4, time_dim=256, text_dim=768).to(device)
    text_encoder = TextEncoder().to(device)
    diffusion = DiffusionModel(unet, timesteps=timesteps)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    tokenizer = SimpleTokenizer()
    
    output_dir = os.path.join(root_folder, "training_results")
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device).float()  # [B, 3, H, W]
            captions = batch['caption']  
            
            latents = torch.cat([images, torch.zeros_like(images[:, :1], dtype=torch.float32)], dim=1)  # [B, 4, H, W]
            
            all_input_ids = []
            all_attention_masks = []
            
            for caption in captions:
                input_ids, attention_mask = tokenizer.encode(caption, device)
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                
            input_ids = torch.cat(all_input_ids, dim=0)
            attention_masks = torch.cat(all_attention_masks, dim=0)
            
            with torch.no_grad():  # Freeze text encoder during training
                text_embeddings = text_encoder(input_ids, attention_masks)
            
            loss = diffusion.train_step(latents, text_embeddings, optimizer)
            epoch_loss += loss
            print(f"Epoch {epoch}, Loss: {loss}")
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"diffusion_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            with torch.no_grad():
                sample_text = "a photograph of plastic bucket"
                sample_ids, sample_mask = tokenizer.encode(sample_text, device)
                sample_embedding = text_encoder(sample_ids, sample_mask)
                
                shape = (1, 4, img_size, img_size)
                sample = diffusion.p_sample_loop(shape, sample_embedding, device)
                
                sample_image = sample[0, :3].permute(1, 2, 0).cpu().numpy()
                sample_image = (sample_image + 1) / 2.0 
                sample_image = np.clip(sample_image, 0, 1)
                
                sample_path = os.path.join(output_dir, f"sample_epoch_{epoch+1}.png")
                sample_pil = Image.fromarray((sample_image * 255).astype(np.uint8))
                sample_pil.save(sample_path)
                print(f"Saved sample image to {sample_path}")
    
    print("Training completed!")
    return unet, text_encoder, diffusion

train()