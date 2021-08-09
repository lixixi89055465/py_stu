import tensorflow as tf
import os

from Resnet18_34 import resnet18
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)

cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)
session = tf.compat.v1.Session(config=config)


def preprocess(x, y):
    # [0，1]
    # [-1，1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# def scheduler(epoch):
#     if epoch < epoch_num * 0.4:
#         return learning_rate
#     if epoch < epoch_num * 0.8:
#         return learning_rate * 0.1
#     return learning_rate * 0.01
#

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
# (x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y=tf.keras.utils.to_categorical(y,100)
y_test = tf.squeeze(y_test, axis=1)
y_test=tf.keras.utils.to_categorical(y_test,100)
print(x.shape, y.shape, x_test.shape, y_test.shape)
## tensorflow2.0 数据增强
print("数据增强:")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                rotation_range=10,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1,
                                                                horizontal_flip=False,
                                                                fill_mode='nearest')
train_datagen.fit(x)
# 测试集不需要进行数据处理
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 不增强验证数据

# train_db = tf.images.Dataset.from_tensor_slices((x, y))
# train_db = train_db.shuffle(10000).map(preprocess).batch(128)
train_db = train_datagen.flow(x, y, batch_size=512)

# test_db = tf.images.Dataset.from_tensor_slices((x_test, y_test))
# test_db = test_db.map(preprocess).batch(128)
# test_db = train_datagen.flow(x_test, y_test, batch_size=512)
test_db = train_datagen.flow(x_test, y_test, batch_size=512)

weight_decay = 5e-4
batch_size = 128
# learning_rate = 1e-2
dropout_rate = 0.5
epoch_num = 200
# change_lr = LearningRateScheduler(scheduler)

model = resnet18()
model.build(input_shape=(None, 32, 32, 3))
model.summary()
# train
optimizer =optimizers.Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_db,  # flows是实时
          steps_per_epoch=x.shape[0] // batch_size,  # steps_per_epoch: 每个epoch所需的steps,不能少
          epochs=200,
          # callbacks=[change_lr],
          validation_data=test_db,
          # validation_data = (test_images, test_labels),
          validation_steps=x_test.shape[0] // batch_size,  # 这个也是不能少
          # callbacks=[tensorboard]
          )
# print(model.summary())
