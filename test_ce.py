import tensorflow as tf

from utils.training import get_valid_logits_and_labels

class_1 = tf.constant([[1, 0], [0, 1]])
class_2 = tf.constant([[0, 0], [1, 0]])
class_3 = tf.constant([[0, 0], [0, 0]])

image = tf.stack([class_1, class_2, class_3], axis=-1)

class_1 = tf.constant([[4, 0], [0, 2]])
class_2 = tf.constant([[0, 0], [4, 0]])
class_3 = tf.constant([[0, 3], [0, 0]])

logits = tf.stack([class_1, class_2, class_3], axis=-1)

image = tf.cast(image, tf.float32)
logits = tf.cast(logits, tf.float32)

old_image = image
image = tf.expand_dims(image, axis=0)
logits = tf.expand_dims(logits, axis=0)

print(image.shape)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=image, logits=logits)

valid_labels, valid_logits = get_valid_logits_and_labels(labels_batch=image, logits_batch=logits)

cross_entropy_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=valid_labels, logits=valid_logits))

unc = tf.where(tf.equal(tf.reduce_sum(image, axis=-1), 0), tf.zeros(shape=(1, 2, 2)), tf.ones(shape=(1, 2, 2)))
loss = tf.losses.compute_weighted_loss(weights = tf.cast(unc, tf.float32),
                                       losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                                           logits = logits,
                                           labels = image))

with tf.Session() as sess:
    print(sess.run(unc))
    print(sess.run(cross_entropy))
    print(sess.run(cross_entropy_valid))
    print(sess.run(loss))