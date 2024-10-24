import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from unet import UNetModel
from diffusion import GaussianDiffusion
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='gray')

device = torch.device('cuda:0')

class FISTA_PnP:
    def __init__(self, unet_model_path, dataset_path, batch_size=1, steps=100, lr=0.01):
        # Load the UNet model
        self.net = UNetModel(image_size=32, in_channels=1, out_channels=1, 
                             model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                             attention_resolutions=[8,4], num_heads=4).to(device)
        self.net.load_state_dict(torch.load(unet_model_path))
        self.net.to(device)
        self.net.eval()  # Set UNet to evaluation mode
        print('Loaded UNet model')

        # Dataset preparation
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(2),
            torchvision.transforms.Normalize(0.5, 0.5),
        ])
        self.dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, 
                                                  transform=self.transforms, download=True)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, 
                                                       shuffle=True, num_workers=4)
        self.steps = steps
        self.lr = lr

    def run(self, img):
        # Initialize variable to optimize (i.e., data consistency step)
        x_fista = nn.Parameter(img.to(device), requires_grad=True)
        opt = torch.optim.Adam([x_fista], lr=self.lr)

        # FISTA parameters
        bar = tqdm.tqdm(range(self.steps))
        t_k = 1
        z_old = x_fista.detach()  # Initialize z_old for FISTA
        y = x_fista.detach()

        # Diffusion process
        diffusion = GaussianDiffusion(T=1000, schedule='linear')

        for i, _ in enumerate(bar):
            # Data consistency step (e.g., gradient descent on data fidelity)
            opt.zero_grad()
            y.grad = None  # Ensure gradients are cleared before optimization
            opt.step()

            # Generate random time steps t
            t = np.random.randint(1, diffusion.T + 1, img.shape[0]).astype(int)  # Random t
            t = torch.from_numpy(t).float().view(img.shape[0])

            # Plug-and-Play regularization step (using pre-trained denoiser)
            with torch.no_grad():
                denoised = self.net(x_fista, t.to(device))  # Denoising using UNet with timesteps

            # FISTA update
            t_k_next = (1 + math.sqrt(1 + 4 * t_k ** 2)) / 2
            y = denoised + ((t_k - 1) / t_k_next) * (denoised - z_old)
            z_old = denoised
            t_k = t_k_next

            x_fista = denoised
        # Return the final reconstructed image

        return x_fista[0,0].detach().cpu().numpy()

if __name__ == "__main__":
    import wandb
    # Path to the UNet model and dataset
    unet_model_path = 'models/mnist_unet.pth'
    dataset_path = 'data/'
    run_name = "Fista_PnP"
    
    # Initialize FISTA_PnP
    fista_pnp = FISTA_PnP(unet_model_path=unet_model_path, dataset_path=dataset_path, batch_size=1, steps=1000, lr=0.01)
    
    wandb.init(project="Course Project", group = "AC=5", name = run_name)
    # Loop through the dataloader and pass the data to the run function
    for img, labels in fista_pnp.data_loader:
        reconstructed_img = fista_pnp.run(img)

        # Ground truth image for comparison (assuming img is the ground truth)
        ground_truth = img[0,0].cpu().numpy()
        
        # Calculate PSNR and SSIM
        psnr = peak_signal_noise_ratio(ground_truth, reconstructed_img, data_range=1.0)
        ssim, _ = structural_similarity(ground_truth, reconstructed_img, full=True, data_range=1.0)

        combined_img = np.hstack((ground_truth, reconstructed_img))

        wandb.log({"PSNR": psnr, "SSIM": ssim, "image": wandb.Image(combined_img)})
