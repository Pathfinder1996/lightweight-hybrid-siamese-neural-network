from keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Flatten, Dense, Input, ReLU, Dropout
)
from keras.models import Model
from keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, EfficientNetB0

from blocks import (
    residual_block, inverted_residual_block, inverted_residual_block_se,
    inverted_residual_block_se_swish, inverted_residual_block_se_relu6,
    se_block, h_swish, h_sigmoid
)

# ResNet18 Architecture
def ResNet18(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    for i in range(len(filters)):
        x = residual_block(x, filters[i], stride=strides[i])
        x = residual_block(x, filters[i], stride=1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="ResNet18")
    model.summary()
    
    return model

# MobileNetV1
def MobileNetV1(input_shape):
    base_model = MobileNet(weights=None, input_shape=input_shape, include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128)(x)
    
    model = Model(inputs=base_model.input, outputs=x, name="MobileNetV1")
    model.summary() 
    
    return model

# MobileNetV2
def MobileNet2(input_shape):
    base_model = MobileNetV2(weights=None, input_shape=input_shape, include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128)(x)
    
    model = Model(inputs=base_model.input, outputs=x, name="MobileNetV2")
    model.summary()
    
    return model

# MobileNetV3Small
def MobileNetV3(input_shape):
    base_model = MobileNetV3Small(weights=None, input_shape=input_shape, include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128)(x)
    
    model = Model(inputs=base_model.input, outputs=x, name="MobileNetV3")
    model.summary()
    return model

# EfficientNetB0
def EfficientNet0(input_shape):
    base_model = EfficientNetB0(weights=None, input_shape=input_shape, include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128)(x)
    
    model = Model(inputs=base_model.input, outputs=x, name="EfficientNetB0")
    model.summary()
    
    return model

# FYO CNN1
def CNN1(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="CNN1")
    model.summary()
    
    return model

# FYO CNN2
def CNN2(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(96, (3,3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(384, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(384, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x) 
    
    x = Flatten()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="CNN2")
    model.summary()
    
    return model

# FYO CNN1 with Squeeze-and-Excitation Block
def CNN1_SE(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="CNN1_SE")
    model.summary()
    
    return model

# FYO CNN2 with Squeeze-and-Excitation Block
def CNN2_SE(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(96, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(384, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(384, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = se_block(x, reduction=16)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="CNN2_SE")
    model.summary()
    
    return model


# ResNet18 + Inverted Residual Block with Squeeze-and-Excitation
def InvResNet18_SE(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 128, 256]
    strides = [1, 2, 2, 2]

    for i in range(4):
        x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="InvResNet18_SE")
    model.summary()
    
    return model

# ResNet18 + Inverted Residual Block with Squeeze-and-Excitation
def InvResNet18_SE_2(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    for i in range(4):
        x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)
    
    x = Dropout(0.3)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="InvResNet18_SE_2")
    model.summary()
    
    return model

def ReLU6(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    for i in range(4):
        x = inverted_residual_block_se_relu6(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="ReLU6")
    model.summary()
    
    return model

def h_swish_network(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    for i in range(4):
        x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="InvResNet18_SE_2")
    model.summary()
    
    return model

# Inverted Residual Block with Squeeze-and-Excitation and h-swish activation
def InvResNet18_SE_Swish(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    for i in range(4):
        x = inverted_residual_block_se_swish(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="InvResNet18_SE_2")
    model.summary()
    
    return model

# Ours paper model architecture
def Ours(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    for i in range(4):
        if i == 0:
            x = inverted_residual_block_se(x, expand_ratio=1, out_channels=filters[i], stride=strides[i], reduction=16)
        else:
            x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="Ours")
    model.summary()
    
    return model

def Ours2(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [32, 64, 128, 256]
    strides = [1, 2, 2, 2]
    for i in range(4):
        if i == 0:
            x = inverted_residual_block_se(x, expand_ratio=1, out_channels=filters[i], stride=strides[i], reduction=16)
        else:
            x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="Ours2")
    model.summary()
    
    return model

def Ours3(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [128, 256, 512, 1024]
    strides = [1, 2, 2, 2]
    for i in range(4):
        if i == 0:
            x = inverted_residual_block_se(x, expand_ratio=1, out_channels=filters[i], stride=strides[i], reduction=16)
        else:
            x = inverted_residual_block_se(x, expand_ratio=6, out_channels=filters[i], stride=strides[i], reduction=16)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="Ours")
    model.summary()
    
    return model

def Ours_without_se_block(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(h_swish)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    for i in range(4):
        if i == 0:
            x = inverted_residual_block(x, expand_ratio=1, out_channels=filters[i], stride=strides[i])
        else:
            x = inverted_residual_block(x, expand_ratio=6, out_channels=filters[i], stride=strides[i])

    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x, name="Ours")
    model.summary()
    
    return model

def get_network(name, input_shape):
    networks = {
        "Ours": Ours,
        "Ours2": Ours2,
        "EfficientNetB0": EfficientNet0,
        "MobileNetV1": MobileNetV1,
        "MobileNetV2": MobileNet2,
        "MobileNetV3": MobileNetV3,
        "CNN1": CNN1,
        "CNN2": CNN2,
        "ResNet18": ResNet18,
        "CNN1_SE": CNN1_SE,
        "CNN2_SE": CNN2_SE,
        "InvResNet18_SE": InvResNet18_SE,
        "InvResNet18_SE_2": InvResNet18_SE_2,
        "InvResNet18_SE_Swish": InvResNet18_SE_Swish,
        "ReLU6": ReLU6,
        "h-swish": h_swish_network,
        "Ours Without SE Block": Ours_without_se_block,
        "Ours3": Ours3,
    }

    if name in networks:
        return networks[name](input_shape)
    else:
        raise ValueError(f"Network {name} not found. Available options: {list(networks.keys())}")
