import os
from typing import Any

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

# ハイパーパラメータ
epochs = 100
batch_size = 64
latent_dim = 128
num_samples = 10000
max_input_len = 330
max_target_len = 330

# データの準備
input_texts = []
target_texts = []

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPUを使用します。")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# データの読み込み
with open('test_3/input0.txt', 'r', encoding='utf-8') as f:
    input_texts = f.read().split('\n')
with open('test_3/output0.txt', 'r', encoding='utf-8') as f:
    target_texts = f.read().split('\n')

debug_max_input_len = 0
debug_max_target_len = 0
input_tokenizer = Tokenizer(oov_token='<OOV>', filters='', lower=False)
# 文字のセットを作成

input_texts = ['<START> ' + text + ' <END>' for text in input_texts]

input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

target_tokenizer = Tokenizer(oov_token='<OOV>', filters='', lower=False)

target_texts = ['<START> ' + text + ' <END>' for text in target_texts]

target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_len,
                                                                   padding='post')
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_len,
                                                                   padding='post')

decoder_target_data = np.zeros((len(target_texts), max_target_len, len(target_tokenizer.word_index) + 1),
                               dtype='float16')
for i, target_sequence in enumerate(target_sequences):
    for t, char in enumerate(target_sequence):
        decoder_target_data[i, t, char] = 1.

encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(len(input_tokenizer.word_index) + 1, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(len(target_tokenizer.word_index) + 1, latent_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

dense_layer_1 = tf.keras.layers.Dense(latent_dim, activation=tf.nn.relu)(decoder_outputs)
decoder_dense = tf.keras.layers.Dense(len(target_tokenizer.word_index) + 1, activation=tf.nn.softmax)

decoder_outputs = decoder_dense(dense_layer_1)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "test_3/training_1/cp{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    model.load_weights(latest)

# model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
#           validation_split=0.2, callbacks=[cp_callback], initial_epoch=0, workers=8, use_multiprocessing=True)

# evaluate, accuracy = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data,
#                                     batch_size=batch_size)
# print("evaluate: ", evaluate)
# print("accuracy: ", accuracy)

encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

reverse_input_char_index = dict((i, char) for char, i in input_tokenizer.word_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_tokenizer.word_index.items())

print(target_tokenizer.word_index.keys())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, batch_size=batch_size, verbose=1)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_tokenizer.word_index['<START>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<END>' or
                len(decoded_sentence) > max_target_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return ''.join(decoded_sentence)


for seq_index in range(50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence_str = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence_str)