import tensorflow as tf
from model.config import cfg

def instead(data):
    return data

def zerof():
    return tf.constant(0.0, dtype=tf.float32)

def concat(a, b):
    return tf.concat([a, b], axis=0)

def masked_det(Ly, mask):
    masked_Ly = tf.boolean_mask(tf.transpose(tf.boolean_mask(Ly, mask)), mask)
    det = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(masked_Ly + cfg.EPSILON * tf.eye(tf.shape(masked_Ly)[0])))))
    return det

def condition(index, summation, label, L):  # batch, obj
    return tf.less(index, tf.shape(label)[0])

def cbody(index, summation, label, L):
    Ly = tf.gather(tf.transpose(tf.gather(L, label[index])), label[index])
    mask = tf.not_equal(Ly, tf.constant(0.0, dtype=tf.float32))[0]
    det = tf.cond(tf.equal(Ly[0][0], tf.constant(0.0, dtype=tf.float32)), lambda: zerof(), lambda: masked_det(Ly, mask))
    return tf.add(index, 1), tf.add(summation, det), label, L

def mbody(index, summation, label, L):
    Lc = tf.gather(tf.transpose(tf.gather(L, label[index])), label[index])
    mask = tf.not_equal(Lc, tf.constant(0.0, dtype=tf.float32))[0]
    masked_Lc = tf.boolean_mask(tf.transpose(tf.boolean_mask(Lc, mask)), mask)
    det = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(masked_Lc + tf.eye(tf.shape(masked_Lc)[0])))))
    return tf.add(index, 1), tf.add(summation, det), label, L

def whilef(params, body):
    return tf.while_loop(condition, body, params)[1]