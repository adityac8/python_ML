def seq2seq(dimx,dimy,num_classes):
    # Recurrent sequence to sequence learning auto encoders for audio classification task  
    print "seq2seq_lstm"
    
    ## encoder
    encoder_input = Input(shape=(dimx,dimy))
    
    encoder=Bidirectional(LSTM(32,return_state=True))# Returns list of nos. of output states
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_input)
    state_h = Concatenate(axis=1)([forward_h, backward_h])
    state_c = Concatenate(axis=1)([forward_c, backward_c])
    encoder_states = [state_h, state_c]
      
    ## decoder
    decoder_input = Input(shape=(dimx,dimy))
    
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_input,
                                         initial_state=encoder_states)
    #h=Flatten()(decoder_outputs)
    decoder_dense = Dense(40, activation='softmax')
    decoder_outputs=decoder_dense(decoder_outputs)
    print kr(decoder_outputs)
    model = Model([encoder_input, decoder_input], decoder_outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])
  
    return model
	
train_x, train_y = load_train_data()
test_x,  test_y  = load_test_data()
target_data=train_x[::-1, :, ::-1]


lrmodel=seq2seq(dimx = train_x.shape[-2], dimy = train_x.shape[-1], num_classes = 15)

lrmodel.fit([train_x,train_x],target_data,batch_size=128,epochs=10,verbose=1)
