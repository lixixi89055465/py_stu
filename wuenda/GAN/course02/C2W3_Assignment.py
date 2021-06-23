import torch
import matplotlib.pyplot as plt


# import torch.nn as nn
# import torch.nn.functional as F

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = torch.nn.functional.make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()


from scipy.stats import truncnorm


def get_truncated_noise(n_sample, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-1 * truncation, truncation, size=(n_sample, z_dim))
    return torch.Tensor(truncated_noise)


assert tuple(get_truncated_noise(n_sample=10, z_dim=5, truncation=0.7).shape) == (10, 5)
simple_noise = get_truncated_noise(n_sample=1000, z_dim=10, truncation=0.2)
assert simple_noise.max() > 0.199 and simple_noise.max() < 2
assert simple_noise.min() < -0.199 and simple_noise.min() > -0.2
assert simple_noise.std() > 0.113 and simple_noise.std() < 0.117

print("Success!")


class MappingLayers(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)

    def get_mapping(self):
        return self.mapping


# Test  the mapping function
map_fn = MappingLayers(10, 20, 30)
a = map_fn(torch.randn(2, 10))
print(a.shape)
assert tuple(a.shape) == (2, 30)
# assert tuple(map_fn(torch.randn(2, 10)).shape) == (2, 30)
assert len(map_fn.mapping) > 4
outputs = map_fn(torch.randn(1000, 10))
assert outputs.std() > 0.05 and outputs.std() < 0.3
assert outputs.min() > -2 and outputs.min() < 0
assert outputs.max() < 2 and outputs.max() > 0
layers=[str(x).replace(' ','').replace('inplace=True','') for x in map_fn.get_mapping()]

assert layers == ['Linear(in_features=10,out_features=20,bias=True)',
                  'ReLU()',
                  'Linear(in_features=20,out_features=20,bias=True)',
                       'ReLU()',
                  'Linear(in_features=20,out_features=30,bias=True)']
print("Success!")