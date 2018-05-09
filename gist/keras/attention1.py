"""
Implemeting Attention on Keras using Theano backend.
"""
def kr(t,m=None):
    if m is None:
        return t._keras_shape
    else:
        return t._keras_shape[m]

def block(inp):
    cnn = Conv2D(128, (3, 3), data_format='channels_first',padding="same", activation="linear", use_bias=False)(inp)
    #cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:,0:64, :, :]

def slice2(x):
    return x[:,64:128,:, :]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],64,input_shape[-2],input_shape[-1]])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],64,input_shape[-2],input_shape[-1]])

def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

def ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    print "ACRNN"
    input_logmel = Input(shape=(1,dimx,dimy))
    a1 = block(input_logmel)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1) 

    a2 = block(a1)
    a2 = block(a2)
    a2 = MaxPooling2D(pool_size=(1,2))(a2)

    a3 = block(a2)
    a3 = block(a3)
    a3 = MaxPooling2D(pool_size=(1,2))(a3) 

    a4 = block(a3)
    a4 = block(a4)
    a4 = MaxPooling2D(pool_size=(1,2))(a4)

    a5 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a4)
    a5 = MaxPooling2D(pool_size=(1,2))(a5) 

    a,b,c,d=kr(a5)
    a6 = Reshape((c*d,b))(a5)

    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='relu', return_sequences=True))(a6)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a6)
    a7 = Multiply()([rnnout, rnnout_gate])

    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a7)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a7)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])

    mymodel = Model([input_logmel], out)
    mymodel.summary()
    
    #opt=optimizers.Adam(1e-3)
    # Compile model
    mymodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['mse'])
    
    return mymodel 
