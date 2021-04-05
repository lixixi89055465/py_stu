import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)

session = tf.compat.v1.Session(config=config)


def preprocess(x, y):
    x = 2. * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
# batchsz = 256
(x, y), (x_val, y_val) = datasets.cifar10.load_data()

y = tf.squeeze(y)
y = tf.one_hot(y, depth=10)
y_val = tf.squeeze(y_val)
y_val = tf.one_hot(y_val, depth=10)
print("x.shape:", x.shape)
print("x_val.shape:", x_val.shape)
print("y.shape:", y.shape)
print("y_val.shape:", y_val.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(60000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

network = Sequential([
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10)
])

network.build(input_shape=(None, 32 * 32 * 3))
print(network.summary())
optimizer = optimizers.Adam(lr=0.0003)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for epoch in range(50):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 32 * 32 * 3))
            out = network(x,training=True)
            # y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))
            loss_meter.update_state(loss)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
        if step % 100 == 0:
            print(step, 'loss:', loss_meter.result().numpy())
            loss_meter.reset_states()
        # evaluate
        if step % 500 == 0:
            total, total_correct = 0., 0
            acc_meter.reset_states()
            for step, (x, y) in enumerate(test_db):
                x = tf.reshape(x, [-1, (32 * 32 * 3)])
                out = network(x,training=False)
                # break
                pred = tf.argmax(out, axis=1)
                y_onehot = tf.argmax(y, axis=1)
                correct = tf.equal(tf.cast(pred, dtype=tf.int64), y_onehot)
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
                acc_meter.update_state(y_onehot, pred)
            print("epoch", epoch, "step", step, "Evaluate Acc:", total_correct / total, acc_meter.result().numpy())
