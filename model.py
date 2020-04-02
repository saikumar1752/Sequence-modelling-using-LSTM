import tensorflow as tf

def build_model(vocab_size, embedding_size, rnn_units, batch_size):
	model=tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_size, batch_input_shape=[batch_size, None]),
		tf.compat.v1.keras.layers.LSTM(
		    rnn_units, 
		    kernel_initializer='glorot_uniform', 
		    recurrent_initializer='glorot_uniform',
		    bias_initializer='glorot_uniform',  
		    kernel_regularizer=tf.keras.regularizers.l2(0.00001),
		    recurrent_regularizer=tf.keras.regularizers.l2(0.00001), 
		    bias_regularizer=tf.keras.regularizers.l1(0.00001), 
		    activity_regularizer=tf.keras.regularizers.l2(0.00001),
		    return_sequences=True, 
		    return_state=False, 
		    stateful=True,
		),
		tf.keras.layers.Dropout(0.2),	
		tf.keras.layers.Dense(vocab_size)
	])
	return model