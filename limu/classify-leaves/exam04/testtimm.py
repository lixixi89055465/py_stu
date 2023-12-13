import torch
import timm

model = timm.create_model('resnet50d')
print(model.default_cfg)  # 查看模型cfg
print('0' * 100)
# TODO timm 案例
model_resnet34 = timm.create_model(
	'resnet34',
	pretrained=True,
	pretrained_cfg_overlay=dict(file='./resnet34_a1_0-46f8f793.pth')
)
# model_resnet34 = timm.create_model('resnet34', pretrained=True)

x = torch.randn([1, 3, 224, 224])
out = model_resnet34(x)
print(out.shape)
# model_list = timm.list_models()#返回一个包含所有模型名称的list
# print(len(model_list))#964
# pretrain_model_list = timm.list_models(pretrained = True)#筛选出带预训练模型的
model_list = timm.list_models()
print(len(model_list))

pretrained_model_list = timm.list_models(pretrained=True)
print('1' * 100)
print(pretrained_model_list)

resnet_model_list = timm.list_models('*resnet*')
print('2' * 100)
print(resnet_model_list)
print('3' * 100)
pretrained_resnet_model_list = timm.list_models('*resnet*', pretrained=True)
print(pretrained_resnet_model_list)
print('4' * 100)
print(list(model_resnet34.children()))
print('5' * 100)
import torch.nn as nn

model = nn.Sequential(*list(model_resnet34.children())[:-1])
print(list(model.children())[-1])
model.add_module('final_layer', nn.Linear(in_features=512, out_features=10, bias=True))
print('6' * 100)
print(list(model.children()))
print('7' * 100)

x = torch.randn([1, 1, 224, 224])
model_resnet34 = timm.create_model(
	'resnet34',
	pretrained=True,
	in_chans=1,
	pretrained_cfg_overlay=dict(file='./resnet34_a1_0-46f8f793.pth'),

)
out = model_resnet34(x)
print(out.shape)
