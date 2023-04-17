import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import os
import io
import tensorflow_datasets as tfds

def test_func():
    # データセットのダウンロードと読み込み
    url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    path = keras.utils.get_file('cornell_movie_dialogs_corpus.zip', origin=url, extract=True)

    # ファイルパス
    movie_lines_filepath = os.path.join(os.path.dirname(path), 'cornell movie-dialogs corpus', 'movie_lines.txt')
    movie_conversations_filepath = os.path.join(os.path.dirname(path), 'cornell movie-dialogs corpus',
                                                'movie_conversations.txt')

    # ファイル読み込み
    with io.open(movie_lines_filepath, 'r', encoding='iso-8859-1') as f:
        movie_lines = f.read().split('\n')
    with io.open(movie_conversations_filepath, 'r', encoding='iso-8859-1') as f:
        movie_conversations = f.read().split('\n')

    # 行と列の数の定義
    num_lines = len(movie_lines) - 1
    num_conversations = len(movie_conversations) - 1

    # データの整形
    # 行のIDをキー、行のテキストを値とする辞書を作成
    id2line = {}
    for line in movie_lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    # 対話IDをキー、対話IDに関連する行のリストを値とする辞書を作成
    conversations = []
    for conversation in movie_conversations:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations.append(_conversation.split(','))

    # 入力とターゲットのリストを作成する
    inputs = []
    targets = []

    # データの整形
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            inputs.append(id2line[conversation[i]])
            targets.append(id2line[conversation[i + 1]])

    # クリーニング関数を定義
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
        return text

    clean_inputs = []
    for input_text in inputs:
        clean_inputs.append(clean_text(input_text))

    clean_targets = []
    for target_text in targets:
        clean_targets.append(clean_text(target_text))

    tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', oov_token="<OOV>")
    tokenizer.fit_on_texts(clean_inputs + clean_targets)
    input_sequences = tokenizer.texts_to_sequences(clean_inputs)
    target_sequences = tokenizer.texts_to_sequences(clean_targets)

    max_sequence_length = 25

    input_padded = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length,
                                                              padding='post', truncating='post')
    target_padded = keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_sequence_length,
                                                               padding='post', truncating='post')

    train_size = int(0.8 * num_lines)

    train_input = input_padded[:train_size]
    train_target = target_padded[:train_size]

    test_input = input_padded[train_size:]
    test_target = target_padded[train_size:]

    encoder_input = keras.layers.Input(shape=(None,))
    encoder_embedding = keras.layers.Embedding(len(tokenizer.word_index) + 1, 128)(encoder_input)
    encoder_gru = keras.layers.GRU(128, return_state=True)
    _, encoder_state = encoder_gru(encoder_embedding)

    decoder_input = keras.layers.Input(shape=(None,))
    decoder_embedding = keras.layers.Embedding(len(tokenizer.word_index) + 1, 128)(decoder_input)
    decoder_gru = keras.layers.GRU(128, return_sequences=True, return_state=True)
    decoder_output, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)
    decoder_dense = keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    model = keras.models.Model([encoder_input, decoder_input], decoder_output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.fit([train_input, train_target[:, :-1]], train_target[:, 1:], epochs=20,
              validation_data=([test_input, test_target[:, :-1]], test_target[:, 1:]))

    encoder_model = keras.models.Model(encoder_input, encoder_state)

    decoder_state_input = keras.layers.Input(shape=(128,))
    decoder_output, decoder_state = decoder_gru(decoder_embedding, initial_state=decoder_state_input)
    decoder_output = decoder_dense(decoder_output)
    decoder_model = keras.models.Model([decoder_input, decoder_state_input], [decoder_output, decoder_state])
    # テストデータからランダムなインデックスを選択
    index = np.random.randint(len(test_input))

    # エンコーダの状態を取得
    state = encoder_model.predict(test_input[index].reshape(1, -1))

    # デコーダの初期入力を設定
    decoder_input = np.zeros((1, 1))
    decoder_input[0, 0] = tokenizer.word_index['<start>']

    # 生成されたテキストを格納するリスト
    generated_text = []

    # テキスト生成ループ
    while True:
        # デコーダの出力と状態を取得
        output, state = decoder_model.predict([decoder_input, state])

        # 最も確率の高い単語を選択
        word_index = np.argmax(output[0, -1, :])
        word = tokenizer.index_word[word_index]

        # 終了条件
        if word == '<end>' or len(generated_text) > max_sequence_length:
            break

        # 生成されたテキストに単語を追加
        generated_text.append(word)

        # 次のデコーダの入力を設定
        decoder_input[0, 0] = word_index

    # 生成されたテキストを文字列に変換

    for i in range(5):
        print("Generated text {}: {}".format(i + 1, ' '.join(generated_text)))

if __name__ == '__main__':
    test_func()