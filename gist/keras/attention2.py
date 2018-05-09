"""
Implemeting Attention on Keras using Theano backend.
"""

def ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=False)(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
    
    #w = Bidirectional(LSTM(32,return_sequences=True))(x)
    rnnout = Bidirectional(LSTM(64, activation='linear', return_sequences=True))(x)
    rnnout_gate = Bidirectional(LSTM(64, activation='sigmoid', return_sequences=True))(x)
    w = Multiply()([rnnout, rnnout_gate])
    
    
    hidden_size = int(w._keras_shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(w)  # hidden_states_t.shape = (batch_size, hidden_size, time_steps)
    
    hidden_states_t = Reshape((hidden_size, hidden_states_t._keras_shape[2]), name='attention_input_reshape')(hidden_states_t)
    # Inside dense layer
    # a (batch_size, hidden_size, time_steps) dot W (time_steps, time_steps) => (batch_size, hidden_size, time_steps)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_states_t._keras_shape[2], use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    #            score_first_part_t         dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('sigmoid', name='attention_weight')(score)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(hidden_size)(a)
    # (batch_size, hidden_size, time_steps) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    main_output = Dense(num_classes, activation='softmax')(attention_vector)
    mymodel = Model(inputs=main_input, outputs=main_output)
    mymodel.summary()
    mymodel.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['mse'])
    return mymodel
