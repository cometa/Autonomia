import keras
import numpy as np
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, PReLU, Dropout, ELU
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2


def coeff_determination(y_true, y_pred):
    '''
    R^2 gives the percentage of the variability between 2 variable that is accounted for. The remaining (1-R^2) is the variability that is not accounted for.
    '''
    SS_res =  K.sum(K.square(y_true - y_pred)) #Residual sum of squares
    SS_tot = K.sum( K.square(y_true - K.mean(y_true)) ) #Residual sum of squares
    coeff = (1 - SS_res/(SS_tot + K.epsilon() ))
    # Return the score
    return coeff


def create_model_2softmax(img_size):
    keep_rate = 0.3
    pool_size = (2, 2)
    
    img_input = Input(shape = img_size)
    x = Convolution2D(16, 5, 5, subsample=(2, 2), W_regularizer=l2(0.001), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 2, 2, subsample=(1, 1), W_regularizer=l2(0.001), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Flatten()(x)
    x = Dropout(keep_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(keep_rate)(x)

    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model

def model_wroscoe(img_size, bucket_sz, keep_rate=0.2, reg_fc=0.01, reg_conv=0.0001, model_type='classification'):
    pool_size = (2, 2)
    num_outputs = bucket_sz
    img_input = Input(shape = img_size)
    x = Convolution2D(img_size[2], 1, 1, subsample=(1, 1), border_mode="same", activation='relu')(img_input)
    x = Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Convolution2D(32, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    x = Dropout(keep_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(keep_rate)(x)

    if model_type == 'classification':
        o_st = Dense(num_outputs, activation='softmax', name='o_st', W_regularizer=l2(reg_fc))(x)
        model = Model(input=img_input, output=o_st)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model is loaded: classification domain ')
    else:
        o_st = Dense(num_outputs, activation='linear', name='o_st', W_regularizer=l2(reg_fc))(x)
        model = Model(input=[img_input], output=[o_st])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        print('Model is loaded: regression domain')
    return model


def model_wroscoe_mod(img_size, bucket_sz, keep_rate=1, reg_fc=0.01, reg_conv=0.0001, model_type='classification'):
    pool_size = (2, 2)
    num_outputs = bucket_sz
    img_input = Input(shape = img_size)
    x = Convolution2D(img_size[2], 1, 1, subsample=(1, 1), border_mode="same", activation='relu', W_regularizer=l2(reg_conv))(img_input)
    #x = Dropout(keep_rate)(x)
    x = Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', W_regularizer=l2(reg_conv))(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    #x = Dropout(keep_rate)(x)
    x = Convolution2D(16, 4, 4, subsample=(1, 1), border_mode="valid", activation='relu', W_regularizer=l2(reg_conv))(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    #x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', W_regularizer=l2(reg_conv))(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    #x = Dropout(keep_rate)(x)
    x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode="valid", activation='relu', W_regularizer=l2(reg_conv))(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    x = Dropout(keep_rate)(x)
    x = Dense(128, activation='relu', W_regularizer=l2(reg_fc))(x)
    x = Dropout(keep_rate)(x)

    if model_type == 'classification':
        o_st = Dense(num_outputs, activation='softmax', name='o_st', W_regularizer=l2(reg_fc))(x)
        model = Model(input=img_input, output=o_st)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model is loaded: classification domain ')
    else:
        o_st = Dense(num_outputs, activation='linear', name='o_st', W_regularizer=l2(reg_fc))(x)
        model = Model(input=[img_input], output=[o_st])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
        print('Model is loaded: regression domain')
    return model



def model_jmlbP3(img_sz, bucket_sz, keep_rate=1, reg_fc=0.01, reg_conv=0.0001, activation_fn = 'relu', model_type='class'):
    '''
    steering angle predictor: takes an image and predict the steerign angle value
    img_sz: size of the image that the model accepts (128, 128, 3)
    activation_fn: non-linear function - relu, prelu or elu
    l2_reg - L2 regularization coefficient for fully connected layers
    ??? NEED TO INCLUDE REG_CONV
    '''

    # size of pooling area for max pooling
    pool_size = (2, 2)
    num_outputs = bucket_sz
    model = Sequential()
    
    model.add(Convolution2D(8, 5, 5, subsample=(1, 1), border_mode="valid", name='conv1', input_shape=img_sz))
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Convolution2D(8, 5, 5, subsample=(1, 1), border_mode="valid") )
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Convolution2D(16, 4, 4, subsample=(1, 1), border_mode="valid") )
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid"))
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))

    model.add(Flatten())
    
    model.add(Dense(128, W_regularizer=l2(reg_fc)))
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(Dense(50, W_regularizer=l2(reg_fc)))
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(Dense(10, W_regularizer=l2(reg_fc)))
    if activation_fn == 'elu':
        model.add(Activation('elu'))
    elif activation_fn == 'prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    if model_type == 'class':
        model.add(Dense(num_outputs, activation='linear', W_regularizer=l2(reg_fc), init='he_normal'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model is loaded: classification domain ')
    else:
        model.add(Dense(num_outputs, activation='linear', W_regularizer=l2(reg_fc), init='he_normal'))
        adam = Adam(lr=0.001) #optimizer
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
        print('Model is loaded: regression domain')
    
    return model


def model_vivekP3(img_size, bucket_sz, keep_rate=1, reg_fc=0.01, reg_conv=0.0001, activation_fn = 'elu', model_type='class'):
    pool_size = (2,2)
    num_outputs = bucket_sz
    filter_size = 3
    model = Sequential()
    model.add(Convolution2D(img_size[2],1,1, border_mode='valid', name='conv0', init='he_normal', input_shape=img_size))
    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid',name='conv1', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid', name='conv2', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(keep_rate))
    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv3', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv4', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(keep_rate))
    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid',name='conv5', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid',name='conv6', init='he_normal', W_regularizer=l2(reg_conv)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(keep_rate))
    model.add(Flatten())
    model.add(Dense(512,name='hidden1', init='he_normal', W_regularizer=l2(reg_fc)))
    model.add(ELU())
    model.add(Dropout(keep_rate))
    model.add(Dense(64,name='hidden2', init='he_normal', W_regularizer=l2(reg_fc)))
    model.add(ELU())
    model.add(Dropout(keep_rate))
    model.add(Dense(16,name='hidden3',init='he_normal', W_regularizer=l2(reg_fc)))
    model.add(ELU())
    model.add(Dropout(keep_rate))
    if model_type == 'classification':
        model.add(Dense(num_outputs, activation='softmax', name='o_st', W_regularizer=l2(reg_fc)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model is loaded: classification domain ')
    else:
        model.add(Dense(num_outputs, activation='linear', name='o_st', W_regularizer=l2(reg_fc)))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        print('Model is loaded: regression domain')
    return model


    '''
    def combined_crossentropy(y_true, y_pred):
    y_true_steering = y_true[:, :num_outputs]
    y_true_throttle = y_true[:, num_outputs:]
    y_pred_steering = y_pred[:, :num_outputs]
    y_pred_throttle = y_pred[:, num_outputs:]

    steering_crossentropy = K.categorical_crossentropy(y_pred_steering, y_true_steering)
    throttle_crossentropy = K.categorical_crossentropy(y_pred_throttle, y_true_throttle)
    return (steering_crossentropy + throttle_crossentropy) / 2.

def create_model_relu2():
    # size of pooling area for max pooling
    pool_size = (2, 2)

    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(256, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Dense(num_outputs, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = RMSprop(lr=0.001)
    model.compile(optimizer=sgd, loss=combined_crossentropy, metrics=['accuracy'])

    return model

def create_model_relu():
    pool_size = (2, 2)
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size)) #added

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Activation('relu'))
#    model.add(Dense(512, init='he_normal'))
    model.add(Dense(256, init='he_normal')) #mod
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(num_outputs, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = RMSprop(lr=0.001)
#    sgd = Adam(lr=0.001)  #mod
    model.compile(optimizer=sgd, loss=combined_crossentropy, metrics=['accuracy'])

    return model

def create_model_2softmax(img_size):
    keep_rate = 0.3
    pool_size = (2, 2)
    img_input = Input(shape = img_size)
    x = Convolution2D(16, 5, 5, subsample=(2, 2), W_regularizer=l2(0.001), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 2, 2, subsample=(1, 1), W_regularizer=l2(0.001), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Flatten()(x)
    x = Dropout(keep_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(keep_rate)(x)

    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model

def create_modelB_2softmax(img_size):
    keep_rate = 0.5
    pool_size = (2, 2)
    img_input = Input(shape = img_size)
    x = Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 5, 5, subsample=(1, 1), border_mode="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    x = Dropout(keep_rate)(x)

    x1 = Dense(900, activation='relu', W_regularizer=l2(0.001))(x)
    x1 = Dropout(keep_rate)(x1)
    x1 = Dense(110, activation='relu', W_regularizer=l2(0.001))(x1)
    x1 = Dropout(keep_rate)(x1)
    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x1)

    x2 = Dense(800, activation='relu', W_regularizer=l2(0.001))(x)
    x2 = Dropout(keep_rate)(x2)
    x2 = Dense(128, activation='relu', W_regularizer=l2(0.001))(x2)
    x2 = Dropout(keep_rate)(x2)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x2)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model
    '''