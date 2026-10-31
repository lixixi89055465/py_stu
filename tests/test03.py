import numpy as np

img_np = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 执行前
print(img_np.shape)  # (3, 640, 480) -> CHW 格式

# 增加 Batch 维度
img_np = np.expand_dims(img_np, axis=0)
print(img_np.shape)
img_np = img_np[np.newaxis, ...]
print(img_np.shape)

#
img_np = img_np[None, ...]
print(img_np.shape)
# 方法3 : reshape

img_np = img_np.reshape(1, *img_np.shape)
print(img_np.shape)
