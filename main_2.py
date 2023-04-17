import MeCab
import gensim
import keras
import numpy as np
from keras.datasets import reuters
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import ndarray
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import transformers

disable_eager_execution()

def test_func():
    embedding_dim = 100  # 埋め込みベクトルの次元数

    tororo = []
    tororo_answer = []

    with open("wordChanger/input/tororo.txt", "r", encoding="utf-8") as f:
        data = f.read()
        for line in data.split("\n"):
            tororo.append(line)

    with open("wordChanger/input/tororo_answer.txt", "r", encoding="utf-8") as f:
        data = f.read()
        for line in data.split("\n"):
            tororo_answer.append(line)

    num_lines = len(tororo) - 1

    id2line = {}
    for i in range(num_lines):
        id2line[i] = tororo[i]

    conversations = []

    for text in tororo_answer:
        conversations.append(text)

    tokenizer = tfds



    wordlist = list(bins.wv.key_to_index)

    vocab_size = len(wordlist)

    word_index = {}
    index_word = {}

    embedding_matrix: ndarray = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(wordlist):
        embedding_vector = bins.wv[word]
        embedding_matrix[i] = embedding_vector
        word_index[word] = i
        index_word[i] = word
        print(str(i) + " / " + str(vocab_size))

    (train_x, train_y), (test_x, test_y) = reuters.load_data(num_words=vocab_size, test_split=0.2)

    num_classes = np.max(train_y) + 1

    print(train_x)

    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        trainable=True)

    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(LSTM(units=50))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(train_x.shape)
    print(train_y.shape)
    # ここからがうまくいかない

    question = gensim.models.Word2Vec.load("wordChanger/output/question/model.bin")
    answer = gensim.models.Word2Vec.load("wordChanger/output/answer/model.bin")
    question_list = list(question.wv.key_to_index)
    answer_list = list(answer.wv.key_to_index)
    q_vocab_size = len(question_list)
    a_vocab_size = len(answer_list)
    question_matrix = np.zeros((q_vocab_size, embedding_dim))
    answer_matrix = np.zeros((a_vocab_size, embedding_dim))

    for i, word in enumerate(question_list):
        print(word)
        embedding_vector = question.wv[word]
        question_matrix[i] = embedding_vector
        word_index[word] = i
        index_word[i] = word
        print(str(i) + " / " + str(q_vocab_size))

    for i, word in enumerate(answer_list):
        print(word)
        embedding_vector = answer.wv[word]
        answer_matrix[i] = embedding_vector
        word_index[word] = i
        index_word[i] = word
        print(str(i) + " / " + str(a_vocab_size))

    question_len = len(question_matrix)
    answer_len = len(answer_matrix)
    if question_len != answer_len:
        if question_len > answer_len:
            new_zeros = np.zeros((len(question_matrix), embedding_dim))
            new_zeros[:answer_matrix.shape[0], :] = answer_matrix
            answer_matrix = new_zeros
        else:
            new_zeros = np.zeros((len(answer_matrix), embedding_dim))
            new_zeros[:question_matrix.shape[0], :] = question_matrix
            question_matrix = new_zeros

    print("aaa")
    print(question_matrix.shape)
    print(answer_matrix.shape)
    model.fit(train_x, train_y, epochs=10, batch_size=128,validation_data=(test_x,test_y),validation_split=0.2) # ここでエラー
    print("bbb")




    def text_to_vector(text, max_sequence_length=200):
        words_list = MeCab.Tagger("-Owakati").parse(text).strip().split(" ")
        vector = np.zeros((1, max_sequence_length))
        for i, word in enumerate(words_list):
            if i >= max_sequence_length:
                break
            if word in word_index:
                vector[0, i] = word_index[word]
        return vector

    # 返答を生成する関数
    def generate_response(text):
        vector = text_to_vector(text)
        prediction = model.predict(vector)
        predicted_word_index = np.argmax(prediction)
        predicted_word = index_word[predicted_word_index]
        return predicted_word

    while True:
        text = input(">> ")
        if text == "exit":
            break
        print(generate_response(text))

if __name__ == '__main__':
    test_func()
