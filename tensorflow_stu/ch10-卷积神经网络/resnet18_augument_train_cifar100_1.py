import tensorflow as tf
import os

from Resnet18_34 import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)

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


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
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
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 不增强验证数据

# train_db = tf.data.Dataset.from_tensor_slices((x, y))
# train_db = train_db.shuffle(10000).map(preprocess).batch(128)
train_db = train_datagen.flow(x, y, batch_size=512)
# test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_db = test_db.map(preprocess).batch(128)
test_db = train_datagen.flow(x_test,y_test,batch_size=512)


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(200):
        correct_sum = 0
        correct_length = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                pred = tf.nn.softmax(logits, axis=1)
                pred = tf.argmax(pred, axis=1)
                correct = tf.equal(pred, tf.cast(y, dtype=tf.int64))
                correct_sum += tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
                correct_length += x.shape[0];
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
        print("test acc is ", float(correct_sum) / float(correct_length))
        total_num = 0
        total_correct = 0.0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)

            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


main()
