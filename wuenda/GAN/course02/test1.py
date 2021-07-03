import torch

# w_prime_prime = w_prime / torch.sqrt(
#     (w_prime ** 2).sum([1, 2, 3])[:, None, None, None] + self.eps
# )

w_prime = torch.ones([3, 3, 3]) * 3
print(w_prime)
print(w_prime ** 2)
print('-' * 100)
print((w_prime ** 2).sum([0, 1]))
