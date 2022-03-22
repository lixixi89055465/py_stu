'''

'''
import torch
input=torch.randn(1,3,64,64)
output_fft_old=torch.rfft(input,signal_ndim=2,normalized=False,onesided=False)
print(output_fft_old.shape)
output_ifft_old=torch.irfft(output_fft_old,signal_ndim=2,normalized=False,onesided=False)
print(output_ifft_old.shape)
#print('1'*100)
# 新版
# input=torch.randn(1,3,64,64)
# output_fft_new=torch.fft.fft2(input,dim=(-2,-1))
# output_fft_new_2dim=torch.stack((output_fft_new.real,output_fft_new.imag), -1)
# print(output_fft_new_2dim.shape)
# output_ifft_new=torch.fft.ifft2(torch.complex(output_fft_new_2dim[...,0],output_fft_new_2dim[...,1]),dim=(-2,-1))
# print(output_ifft_new.shape)
# output_ifft_new=torch.fft.ifft2(output_fft_new_2dim,dim=(-2,-1))
# print(output_ifft_new.shape)

