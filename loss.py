import tensorflow as tf

# NOTE: this loss extracted from the following tensorflow tutorial https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
