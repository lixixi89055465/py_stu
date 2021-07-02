import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=2)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

class ModulatedConv2d(nn.Module):
    '''
    ModulatedConv2d Class, extends/subclass of nn.Module
    Values:
      channels: the number of channels the image has, a scalar
      w_dim: the dimension of the intermediate tensor, w, a scalar
    '''

    def __init__(self, w_dim, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style_scale_transform = nn.Linear(w_dim, in_channels)
        self.eps = 1e-6
        self.padding = padding

    def forward(self,image,w):
        images=[]
        for i,w_cur in enumerate(w):
            style_scale=self.style_scale_transform(w_cur)
            w_prime=self.conv_weight*style_scale[None,:,None,None]
            w_prime_prime=w_prime/torch.sqrt(
                (w_prime**2).sum([1,2,3])[:,None,None,None]+self.eps
            )
            images.append(F.conv2d(image[i][None,:,:,:],w_prime_prime,padding=self.paddig))
        return torch.cat(images)

example_modulated_conv = ModulatedConv2d(w_dim=128, in_channels=3, out_channels=3, kernel_size=3)
num_ex = 2
image_size = 64
rand_image=torch.randn(num_ex,3,image_size,image_size)
rand_w=torch.randn(num_ex,128)
new_image=example_modulated_conv(rand_image,rand_w)
second_modulated_conv=ModulatedConv2d(w_dim=128,in_channels=3,out_channels=3,kernel_size=3)
second_image=second_modulated_conv(new_image,rand_w)

print("Original noise (left), noise after modulated convolution (middle), noise after two modulated convolutions (right)")
plt.rcParams['figure.figsize'] = [8, 8]
show_tensor_images(torch.stack([rand_image, new_image, second_image], 1).view(-1, 3, image_size, image_size))