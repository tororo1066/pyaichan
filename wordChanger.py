import MeCab
from gensim.models import word2vec


def model_func():
    # 学習データのパス
    data_path = "wordChanger/input/tororo_answer.txt"

    # 学習データの読み込み
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read()

    print("debug0")

    # 形態素解析器の初期化
    tagger = MeCab.Tagger("-Owakati")

    print("debug1")

    sentences = []

    length = len(data.split("\n"))

    for index, line in enumerate(data.split("\n")):
        sentences.append(tagger.parse(line).strip().split(" "))
        print(str(index) + " / " + str(length))

    # Word2Vecの学習
    model = word2vec.Word2Vec(sentences, window=5, min_count=1, workers=4, vector_size=100)

    # モデルの保存
    model.save("wordChanger/output/model.bin")


if __name__ == '__main__':
    model_func()
