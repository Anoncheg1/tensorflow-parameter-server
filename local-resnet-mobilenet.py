import tensorflow as tf
import numpy as np
import os
# import importlib

# -- model mapping "model name" for imclassif.py
select_model = {"resnet": "ResNet",
                "mobilenet": "MobileNet"}
# SELECTED_MODEL="resnet"
SELECTED_MODEL="mobilenet"
os.environ['MODEL_NAME'] = select_model[SELECTED_MODEL]
import imclassif as m # model, train_dataset, x_valid, y_valid, BATCH_SIZE, encode_single_sample, class_weight

# ---- Dataset instances ----
train_dataset = m.train_dataset.batch(m.BATCH_SIZE).prefetch(100)
validation_dataset = tf.data.Dataset.from_tensor_slices((m.x_valid.astype(str), m.y_valid.astype(int))).skip(m.x_valid.shape[0] - m.x_valid.shape[0]//4)
validation_dataset = validation_dataset.map(lambda x, y: m.encode_single_sample(x, y), tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.batch(m.BATCH_SIZE).prefetch(100)

# ---- train ----
m.model.fit(train_dataset, class_weight=m.class_weight, epochs=1)

# -- checks the model's performance
print("evaluate")
m.model.evaluate(validation_dataset, verbose=2)

# -- inferece
print("inference", m.x_valid[0], m.y_valid[0])
im, l = m.encode_single_sample(m.x_valid[0], m.y_valid[0])
im = tf.expand_dims(im, axis=0)
print("im", im.shape)
predictions = m.model.predict(im, batch_size=1)
print(np.argmax(predictions))
print("label:", m.y_valid[0])
