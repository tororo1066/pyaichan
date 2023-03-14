import re
import xml.etree.ElementTree
from bs4 import BeautifulSoup


def wiki2text():
    # 日本語WikipediaコーパスのXMLファイルを読み込む
    tree = xml.etree.ElementTree.parse('wiki2text/input/jawiki-latest-pages-articles.xml')
    itre = tree.getroot().iter('{http://www.mediawiki.org/xml/export-0.10/}page')

    # ページのタイトルと本文を取得する
    for index, page in enumerate(itre):
        text = page.find(
            '{http://www.mediawiki.org/xml/export-0.10/}revision').find(
            '{http://www.mediawiki.org/xml/export-0.10/}text').text

        if text is None:
            continue

        text = BeautifulSoup(text, 'html.parser').get_text()
        # 本文から不要な部分を除去する
        text = text.replace('\r', '').replace('\t', '')

        # 処理したテキストデータを保存するなどして利用する
        # ...

        print(index)

        with open("wordChanger/input/jawiki.txt", "a", encoding="utf-8") as f:
            f.write(text)


def split_file():

    chunk_size = 800 * 1024 * 1024  # 100MB

    with open('wordChanger/input/jawiki.txt', 'rb') as f:
        chunk_id = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f'wordChanger/input/chunk_{chunk_id}.txt', 'wb') as out_file:
                out_file.write(chunk)
            chunk_id += 1


if __name__ == '__main__':
    split_file()
