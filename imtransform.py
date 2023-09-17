import tensorflow as tf
import math
import numpy as np

def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 360.
    shear    = math.pi * shear    / 360.

    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')

    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    rot_mat = get_3x3_mat([c1,    s1,   zero ,
                           -s1,   c1,   zero ,
                           zero,  zero, one ])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_mat = get_3x3_mat([one,  s2,   zero ,
                             zero, c2,   zero ,
                             zero, zero, one ])

    zoom_mat = get_3x3_mat([one/hzoom, zero,      zero,
                            zero,      one/wzoom, zero,
                            zero,      zero,      one])

    shift_mat = get_3x3_mat([one,  zero, hshift,
                             zero, one,  wshift,
                             zero, zero, one   ])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )


def spatial_transform(image,
                       rotation=3.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):

    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # origin pixels
    y = tf.repeat(tf.range(ydim//2, -ydim//2,-1), xdim)
    x = tf.tile(tf.range(-xdim//2, xdim//2), [ydim])
    z = tf.ones([ydim*xdim], dtype='int32')
    idx = tf.stack([y, x, z])

    # destination pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    # clip to origin pixels range
    idx2y = tf.clip_by_value(idx2[0,], -ydim//2+yxdim+1, ydim//2)
    idx2x = tf.clip_by_value(idx2[1,], -xdim//2+xxdim+1, xdim//2)
    idx2 = tf.stack([idx2y, idx2x, idx2[2,]])

    # apply destinations pixels to image
    idx3 = tf.stack([ydim//2-idx2[0,], xdim//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    image = tf.reshape(d, [ydim, xdim, 3])
    return image

def pixel_transform(image,
                     saturation_delta=0.3,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    image = tf.image.random_saturation(
        image, 1-saturation_delta, 1+saturation_delta)
    image = tf.image.random_contrast(
        image, 1-contrast_delta, 1+contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image

def oversample(xy, maxc=None):
    unq, unq_idx = np.unique(xy[:, -1], return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    if maxc:
        cnt = maxc
    else:
        cnt = np.max(unq_cnt)
    out = np.empty((cnt*len(unq) - len(xy),) + xy.shape[1:], xy.dtype)
    slices = np.concatenate(([0], np.cumsum(cnt - unq_cnt)))
    for j in range(len(unq)):
        indices = np.random.choice(np.where(unq_idx==j)[0], cnt - unq_cnt[j])
        out[slices[j]:slices[j+1]] = xy[indices]
    return np.vstack((xy, out))

def oversamples_half(xy):
    # - separate part of xy with classes which count of examples > max(count of examples)//2
    unq, unq_idx = np.unique(xy[:, -1].astype(int), return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    cnt_half = np.max(unq_cnt) //2
    use_u = unq[unq_cnt<cnt_half]
    use_i = np.vectorize(lambda x: x in use_u)(xy[:,-1])
    use = xy[use_i]
    not_use = xy[~use_i]
    # print("use", np.bincount(use[:,1].astype(int)))
    out = oversample(use, maxc=cnt_half)
    # print("out", np.bincount(out[:,1].astype(int)))
    return np.vstack((out, not_use))

def calc_class_weights(y_train):
    """ for [0]*5 + [1]*2 + [2]*5
        :return {0: 0.8, 1: 2.0, 2: 0.8}
    """
    classes = sorted(set(y_train))
    n_classes = len(classes)
    n_samples = len(y_train)
    y_train = y_train.astype(int)
    weights = n_samples / (n_classes * np.bincount(y_train))
    return {c:w for c,w in zip(classes, weights)}
