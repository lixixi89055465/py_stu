import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers, Sequential,metrics
import datetime
from matplotlib import pyplot as plt
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def plot_to_image(figure):
    buf=io.BytesIO()
    plt.savefig(buf,format='png')
    plt.close(figure )
    buf.seek(0)





(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape)
print(y.shape)
batchsz = 128
# batchsz = 16
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print(type(sample))
print(sample[0].shape)
print(sample[1].shape)
model = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=(None, 28 * 28))
# print(model.summary())
optimizer = optimizers.Adam(lr=1e-3)


# print(model.trainable_variables)


def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)
            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, float(loss_ce), float(loss_mse))
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            # correct = tf.equal(tf.cast(pred,dtype=tf.int32), y)
            correct = tf.equal(tf.cast(pred, dtype=tf.int32), y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += correct
            total_num += x.shape[0]
        acc = total_correct / total_num
        print(epoch, 'test ac :', acc)


if __name__ == '__main__':
    main()
