import joblib
from tensorflow import keras
import os
print(os.path.abspath("."))
a = joblib.load("scores/german/german_model.h5.score")
print(a)

# model = keras.models.load_model("models/adult.model.h5")
# model.summary()
#
# model = keras.models.load_model('models/bank_model.h5')
# model.summary()
#
import tensorflow as tf
from tensorflow.keras import backend as K
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = K.variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, min, max))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

model = keras.models.load_model("models/repaired_models/race_gated_layer4_per0.3_thresh0.2_va0.821_ra0.993.h5", custom_objects={'ScaleLayer': ScaleLayer})
model.summary()