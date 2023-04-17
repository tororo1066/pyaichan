import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

# データセットをダウンロード
train_data, test_data = tfds.load("wikipedia/20230201.ja", split='train', shuffle_files=True, with_info=True)


# 入力データをトークン化
# tokenizer = tfds.deprecated.text.Tokenizer()
# vocabulary_set = set()
# for text_tensor, _ in train_data:
#     print(text_tensor, _)
#     some_tokens = tokenizer.tokenize(text_tensor.numpy())
#     vocabulary_set.update(some_tokens)
#
# # 入力データをエンコード
# encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
#
#
# def encode(text_tensor, label):
#     encoded_text = encoder.aencode(text_tensor.numpy())
#     return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((text['text'].numpy() for text in train_data),
                                                                      target_vocab_size=2 ** 13)


def encode(text_tensor, label):
    encode_text = tokenizer.encode(text_tensor.numpy())
    return encode_text, label


train_data = train_data.map(lambda key, value: encode(key['text'], value['title']))

train_dataset = train_data.shuffle(10000).padded_batch(64, padded_shapes=([-1], []))

# モデルを構築
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6)
])

# モデルをコンパイル
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# モデルをトレーニング
model.fit(train_dataset,
          epochs=3,
          validation_data=test_data.batch(32),
          verbose=1)

# モデルを評価
# テストデータに対する予測結果を取得
test_loss, test_acc = model.evaluate(test_data.batch(32), verbose=2)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# テストデータに対する予測結果を出力
for text, label in test_data.take(5):
    encoded_text, _ = encode(text, label)
    prediction = model.predict(tf.expand_dims(encoded_text, 0))
    print('Input text: {}'.format(text.numpy()))
    print('Label: {}'.format(label.numpy()))
    print('Prediction: {}'.format(tf.argmax(prediction, axis=1).numpy()))

if __name__ == '__main__':
    pass
