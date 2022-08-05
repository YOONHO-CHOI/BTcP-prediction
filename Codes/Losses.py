#%% Import modules
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


#%% Define loss functions
def Weighted_BCE(weights):
    def wbce(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true)*K.binary_crossentropy(y_true, y_pred)), axis=-1)
    return wbce

def Dice_BCE(weights):
    # flatten label and prediction tensors
    def dicebce(y_true, y_pred, smooth=1e-6):
        inputs = tf.keras.backend.flatten(y_pred)
        targets = tf.keras.backend.flatten(y_true)
        BCE = K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true)*K.binary_crossentropy(y_true, y_pred)), axis=-1)
        intersection = tf.keras.backend.sum(targets* inputs)
        dice_loss = 1 - (2 * intersection + smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + smooth)
        Dice_BCE = 1*BCE + 0*dice_loss
        return Dice_BCE
    return dicebce

def MCC_multiclass(false_pos_penal=1.0):
    def mcc_value(y_true, y_pred, false_pos_penal=false_pos_penal):
        confusion_m = tf.matmul(K.transpose(y_true), y_pred)
        if false_pos_penal != 1.0:
            """
            This part is done for penalization of FalsePos symmetrically with FalseNeg,
            i.e. FalseNeg is favorized for the same factor. In such way MCC values are comparable.
            If you want to penalize FalseNeg, than just set false_pos_penal < 1.0 ;)
            """
            confusion_m = tf.matrix_band_part(confusion_m, 0, 0) + tf.matrix_band_part(confusion_m, 0, -1) * false_pos_penal + tf.matrix_band_part(confusion_m, -1, 0) / false_pos_penal
        N = K.sum(confusion_m)
        up = N * tf.linalg.trace(confusion_m) - K.sum(tf.matmul(confusion_m, confusion_m))
        down_left = K.sqrt(N ** 2 - K.sum(tf.matmul(confusion_m, K.transpose(confusion_m))))
        down_right = K.sqrt(N ** 2 - K.sum(tf.matmul(K.transpose(confusion_m), confusion_m)))
        mcc = up / (down_left * down_right + K.epsilon())
        return mcc
    return mcc_value

def MCC():
    def mcc_(y_true, y_pred):
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0) * 1e2
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0) / 1e2
        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = up / (down + K.epsilon())
        mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)
        mcc = 1 - K.mean(mcc)
        return mcc
    return mcc_

def Balanced_CE():
    def Balce(y_true, y_pred):
        beta = tf.reduce_mean(1-y_true)
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)
    return Balce

def Weighted_CE():
    def loss(y_true, y_pred):
        beta = tf.reduce_mean(1 - y_true)
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)
    return loss

def Focal(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)
    return loss

def Tversky(beta=0.3):
  def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
  return tversky

def Tversky_MSE(beta, gamma):
  def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    tversky = 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
    mse = tf.keras.losses.MeanSquaredError(y_true, y_pred)
    return (1-gamma)*tversky + gamma*mse
  return loss
