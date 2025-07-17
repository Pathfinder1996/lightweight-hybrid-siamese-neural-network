import tensorflow as tf
import keras.backend as K
from keras.layers import (
    Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, ReLU
)

# hard swish and hard sigmoid functions
def h_swish(x):
    return x * K.relu(x + 3, max_value=6) / 6

def h_sigmoid(x):
    return K.relu(x + 3, max_value=6) / 6

# BatchNormalization + ReLU
def batch_norm_act(x, act=True):
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)

    return x

# Convolutional block
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = batch_norm_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

    return conv

# Residual block for ResNet18
def residual_block(x, filters, stride=1):
    shortcut = x

    out = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding="same", 
                 kernel_initializer='he_normal')(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)

    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same", 
                 kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)

    if stride != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, 
                          kernel_initializer='he_normal', padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = Add()([out, shortcut])
    out = Activation("relu")(out)

    return out

# def se_block(input_tensor, reduction=16):
#     # Squeeze-and-Excitation Block
#     channels = input_tensor.shape[-1]
#     se = GlobalAveragePooling2D()(input_tensor)   # shape=(batch, C)
#     se = Dense(units=channels // reduction, activation='relu')(se)
#     se = Dense(units=channels, activation='sigmoid')(se)
#     se = tf.reshape(se, [-1, 1, 1, channels])  # shape=(batch,1,1,C)
#     output = input_tensor * se

#     return output

def inverted_residual_block(x, expand_ratio, out_channels, stride=1):
    in_channels = x.shape[-1]

    # 1. Pointwise Conv (Expand)
    expanded_channels = in_channels * expand_ratio
    out = Conv2D(expanded_channels, kernel_size=(1,1), padding='same', use_bias=False)(x)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)

    # 2. Depthwise 3×3 Conv
    out = DepthwiseConv2D(kernel_size=(3,3), strides=stride, padding='same', use_bias=False)(out)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)

    # 3. Pointwise Conv (Project)
    out = Conv2D(out_channels, kernel_size=(1,1), padding='same', use_bias=False)(out)
    out = BatchNormalization()(out)

    # 4. Shortcut
    if stride == 1 and in_channels == out_channels:
        out = Add()([x, out])   

    return out

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // reduction)(se)
    se = ReLU(max_value=6)(se)
    se = Dense(filters)(se)
    se = h_sigmoid(se)
    
    return input_tensor * tf.expand_dims(tf.expand_dims(se, 1), 1)

def inverted_residual_block_se(x, expand_ratio, out_channels, stride=1, reduction=16):
    in_channels = x.shape[-1]
    shortcut = x

    # 1. 1x1 Expand
    hidden_dim = in_channels * expand_ratio
    out = Conv2D(hidden_dim, kernel_size=1, padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)

    # 2. 3x3 Depthwise Convolution
    out = DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)

    # 3. Squeeze-and-Excitation block
    out = se_block(out, reduction=reduction)

    # 4. 1x1 project
    out = Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)

    if stride == 1 and in_channels == out_channels:
        out = Add()([shortcut, out])
        
    return out

def inverted_residual_block_se_relu6(x, expand_ratio, out_channels, stride=1, reduction=16):
    in_channels = x.shape[-1]
    shortcut = x

    # 1. 1x1 Expand
    hidden_dim = in_channels * expand_ratio
    out = Conv2D(hidden_dim, kernel_size=1, padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)

    # 2. 3x3 Depthwise Convolution
    out = DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)
    out = ReLU(max_value=6)(out)
    
    # 3. Squeeze-and-Excitation Block
    out = se_block(out, reduction=reduction)

    # 4. 1x1 Project
    out = Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)

    # Shortcut
    if stride == 1 and in_channels == out_channels:
        out = Add()([shortcut, out])

    return out

def inverted_residual_block_se_swish(x, expand_ratio, out_channels, stride=1, reduction=16):
    in_channels = x.shape[-1]
    shortcut = x

    hidden_dim = in_channels * expand_ratio
    out = Conv2D(hidden_dim, kernel_size=1, padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    if in_channels > 128:
        out = Activation(h_swish)(out)
    else:
        out = ReLU(max_value=6)(out)(out)

    out = DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)
    out = Activation(h_swish)(out)
    
    out = se_block(out, reduction=reduction)

    out = Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)

    if stride == 1 and in_channels == out_channels:
        out = Add()([shortcut, out])

    return out
