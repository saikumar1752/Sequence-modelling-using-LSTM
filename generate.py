import tensorflow as tf
import numpy as np
import model
import sys

print(sys.argv)

char2int=np.load('char2int.npy', allow_pickle=True)
int2char=np.load('int2char.npy', allow_pickle=True)
text_as_int=np.load('text_as_int.npy', allow_pickle=True)
vocab_size=np.load('vocab_size.npy', allow_pickle=True)

SEQ_LENGTH=150
BATCH_SIZE=64
BUFFER_SIZE=100000
vocab_size=vocab_size[0]
embedding_size=256
rnn_units=1024


model = model.build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_sequence, num_generate):
	input_eval=[char2idx[s] for x in start_sequence]
	input_eval=tf.expand_dims(input_eval, 0)

	text_generated=[]

	temperature=0.6

	model.reset_states()
	for i in range(num_generate):
		predictions=model(input_eval)
		predictions=tf.squeeze(predictions, 0)
		predictions=prediction/temperature
		predicted_id=tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
		input_eval=tf.expand_dims([predicted_id], 0)
		text_generated.append(int2char[predicted_id])
	return text_generated

start_sequence=sys.argv[1]
length_of_text=sys.argv[2]

print(generate_text(model, start_sequence, length_of_text))