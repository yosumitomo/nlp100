# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:56:43 2018

@author: owner

NLP 100 knock
http://www.cl.ecei.tohoku.ac.jp/nlp100/

"""
# %% import
import re
import os
import MeCab
import numpy as np
import json
from itertools import compress
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pydot
import subprocess

# %% 第1章: 準備運動


# %% 00. 文字列の逆順
# 文字列"stressed"の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．
def ans_00():
    str = "strssed"
    print(str[::-1])

# %% 01. 「パタトクカシーー」
# 「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．
def ans_01():
    str = "パタトクカシーー"
    print(str[0:8:2])

# %% 02. 「パトカー」＋「タクシー」＝「パタトクカシーー」
# 「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．
def ans_02():
    ptk = "パトカー"
    taxi = "タクシー"
    ret = ""
    for p, t in zip(ptk, taxi):
        ret += p + t
    print(ret)

# %% 03. 円周率
# "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
# という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．
def ans_03():
    vec = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    words = vec.replace(",","").replace(".","").split(" ")
    m = list(map(lambda x: len(x), words))
    print(m)

    # mecabで何かできないか遊んでた時のもの
    '''
    import MeCab
    m = MeCab.Tagger()
    print(m.parse(vec))
    '''

# %% 04. 元素記号
# "Hi He Lied Because Boron Could Not Oxidize Fluorine.
# New Nations Might Also Sign Peace Security Clause.
# Arthur King Can."という文を単語に分解し，
# 1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，
# それ以外の単語は先頭に2文字を取り出し，
# 取り出した文字列から単語の位置（先頭から何番目の単語か）への
# 連想配列（辞書型もしくはマップ型）を作成せよ．

def ans_04():
    words = "Hi He Lied Because Boron Could Not \
    Oxidize Fluorine. New Nations Might Also Sign \
    Peace Security Clause. Arthur King Can."
    words = re.sub('[,.]', '', words).split(" ")
    ret = {}
    for i in range(0, len(words), 2):
        ret.update({words[i][0]: i+1})
    for i in range(1, len(words), 2):
        ret.update({words[i][0:2]: i+1})


# %% 05. n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
# この関数を用い，"I am an NLPer"という文から単語bi-gram，文字bi-gramを得よ．
def n_gram_05(seq, n, opt):
    ret = []
    if opt == 'words':
        words = re.sub('[,.]', '', seq).split(" ")
        for i in range(0, len(words)-(n-1)):
            ret.append(words[i] + " " + words[i+(n-1)])
    elif opt == 'letter':
        for i in range(0, len(seq)-(n-1)):
            ret.append(seq[i:i+n])
    return ret


def ans_05():
    seq = "I am an NLPer"
    opt = ['words', 'letter']
    n = 2
    for i in range(0, len(opt)):
        print(n_gram_05(seq, n, opt=opt[i]))
    # print([n_gram_05(seq, n, x) for x in opt])
    # print(list(map(lambda x: n_gram_05(seq, n, x), opt)))


# %% 06. 集合
# "paraparaparadise"と"paragraph"に含まれる文字bi-gramの集合を，
# それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
# さらに，'se'というbi-gramがXおよびYに含まれるかどうかを調べよ．
def ans_06():
    set_names = ['X', 'Y']
    seq_x = "paraparaparadise"
    seq_y = "paragraph"
    X = set(n_gram_05(seq_x, 2, 'letter'))
    Y = set(n_gram_05(seq_y, 2, 'letter'))
    set_union = X | Y  # 和集合
    set_intersection = X & Y  # 積集合
    set_difference = X - Y  # 差集合

    print(set_union)
    print(set_intersection)
    print(set_difference)

    i = 0
    for curr_set in [X, Y]:
        query = ['se']
        flag = len(curr_set & set(query))
        if flag:
            print('"' + query[0] + '" is contained in ' + set_names[i])
        else:
            print('"' + query[0] + '" is NOT contained in ' + set_names[i])
        i += 1


# %% 07. テンプレートによる文生成
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．
# さらに，x=12, y="気温", z=22.4として，実行結果を確認せよ．
def concat_07(x, y, z):
    return str(x) + '時の' + str(y) + 'は' + str(z)


def ans_07():
    x = 12
    y = "気温"
    z = 22.4
    ret = concat_07(x, y, z)
    print(ret)


# %% 08. 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
#
# 英小文字ならば(219 - 文字コード)の文字に置換
# その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．
def cipher_08(seq):
    letter = []
    for i in seq:
        if i.islower():
            letter.append(chr(219 - ord(i)))
        else:
            letter.append(i)
    ret = ''.join(letter)
    return ret


def ans_08():
    seq = '12abABcd=~ef'
    print(seq)
    print(cipher_08(seq))
    print(cipher_08(cipher_08(seq)))


# %% 09. Typoglycemia
# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，
# それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文（例えば
# "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."）
# を与え，その実行結果を確認せよ．
def randomize_09(seq):
    words = seq.split(" ")
    ret = ""
    for i in words:
        print(len(i))
        if len(i) > 4:
            curr = " " + i[0]
            idx = np.random.permutation(list(range(1, len(i)-1)))
            for j in idx:
                curr = curr + i[j]
            curr = curr + i[-1]
            ret = ret + curr
        else:
            ret = ret + " " + i
    return ret


def ans_09():
    seq = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    ret = randomize_09(seq)
    print(ret)


# %% 第2章: UNIXコマンドの基礎
# hightemp.txtは，日本の最高気温の記録を「都道府県」「地点」「℃」「日」のタブ区切り形式で格納したファイルである．
# 以下の処理を行うプログラムを作成し，hightemp.txtを入力ファイルとして実行せよ．
# さらに，同様の処理をUNIXコマンドでも実行し，プログラムの実行結果を確認せよ．


# %% 10. 行数のカウント
# 行数をカウントせよ．確認にはwcコマンドを用いよ．
def ans_10():
    l = 0;
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            l += 1
    print(l)


# %% 11 . タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．
# 確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．


def ans_11():
    rows = []
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            rows.append(line)
    rows = list(map(lambda x: x.replace("\t", ' '), rows))
    print(rows)


# %% 12. 1列目をcol1.txtに，2列目をcol2.txtに保存
# 各行の1列目だけを抜き出したものをcol1.txtに，
# 2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
# 確認にはcutコマンドを用いよ．
def ans_12():
    rows = []
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            rows.append(line)
    col1 = []
    col2 = []
    for row in rows:
        row = row.split("\t")
        col1.append(row[0])
        col2.append(row[1])

    with open('col1.txt', mode='w', encoding='utf-8') as f:
        for x in col1:
            f.write(str(x) + "\n")
    with open('col2.txt', mode='w', encoding='utf-8') as f:
        for x in col2:
            f.write(str(x) + "\n")

# %% 13. col1.txtとcol2.txtをマージ
# 12で作ったcol1.txtとcol2.txtを結合し，
# 元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
# 確認にはpasteコマンドを用いよ．
def ans_13():
    col1 = []
    col2 = []
    with open('col1.txt', encoding='utf-8') as f:
        for line in f:
            col1.append(line)
    with open('col2.txt', encoding='utf-8') as f:
        for line in f:
            col2.append(line)

    col1 = list(map(lambda x: re.sub("\n", "", x), col1))
    col2 = list(map(lambda x: re.sub("\n", "", x), col2))

    merged = []
    for x, y in zip(col1, col2):
        merged.append(x + "\t" + y)

    with open('merged.txt', mode='w', encoding='utf-8') as f:
        for x in merged:
            f.write(str(x) + "\n")


# %% 14. 先頭からN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，
# 入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ

def ans_14():
    i = 0
    n = 5
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            if i > n-1:
                break
            print(re.sub("\n", "", line))
            i += 1


# %% 15. 末尾のN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，
# 入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．
def ans_15():
    i = 0
    n = 5
    rows = [''] * n
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            rows[i] = line.replace("\n","")
            print(rows)
            i += 1
            i = i % n
    for j in range(i, i+n):
        print(rows[j % n])


# %% 16. ファイルをN分割する
# 自然数Nをコマンドライン引数などの手段で受け取り，
# 入力のファイルを行単位でN分割せよ．
# 同様の処理をsplitコマンドで実現せよ．
def ans_16():
    path = './split_python/'
    if not os.path.exists(path):
        os.makedirs(path)

    i = 0
    file_counts = 0
    n = 5
    rows = []
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            rows.append(line.replace("\n", ""))
            i += 1
            i = i % n
            if i == 0:
                with open(path + str(file_counts).zfill(2) + '.txt',
                          mode='w', encoding='utf-8') as f:
                    for x in rows:
                        f.write(x + "\n")
                file_counts += 1
                rows = []
        if i != 0:
            with open(path + str(file_counts).zfill(2) + '.txt',
                      mode='w', encoding='utf-8') as f:
                for x in rows:
                    f.write(x + "\n")

# キモイ

# %% 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはsort, uniqコマンドを用いよ．
def ans_17():
    words = set()
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            words.add(line.split("\t")[0])
    print(words)

# %% 18. 各行を3コラム目の数値の降順にソート
# 各行を3コラム目の数値の逆順で整列せよ
# （注意: 各行の内容は変更せずに並び替えよ）．
# 確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．
def ans_18():
    rows = []
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            rows.append(line)
    tempreture = np.empty(len(rows))
    for i in range(len(rows)):
        tempreture[i] = float(rows[i].split("\t")[2])
    idx = np.argsort(tempreture)
    ret = [''] * len(idx)
    for i in range(len(idx)):
        ret[i] = rows[idx[i]]
    for curr in ret:
        print(curr.replace("\n", ""))


# %% 19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
# 各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
# 確認にはcut, uniq, sortコマンドを用いよ.
def ans_19():
    pref = []
    with open('hightemp.txt', encoding='utf-8') as f:
        for line in f:
            pref.append(line.split("\t")[0])
    pref_uniq = list(set(pref))
    num_duplicate = []
    for key in pref_uniq:
        num_duplicate.append(sum(map(lambda x: x == key, pref)))
    idx = np.argsort(num_duplicate)
    # pref_sorted_inv_by_num_duplicate = []
    for i in idx[::-1]:
        curr_pref = pref_uniq[i]
        num = sum(map(lambda x: x == curr_pref, pref))
        print("\t" + str(num) + "\t" + curr_pref)
        # pref_sorted_inv_by_num_duplicate.append(pref_uniq[i])
    # print(pref_sorted_inv_by_num_duplicate)

    # なんか処理が重複してる気がするけどまあいいわ


# %% 第3章: 正規表現
# Wikipediaの記事を以下のフォーマットで書き出したファイルjawiki-country.json.gzがある．
#
# 1行に1記事の情報がJSON形式で格納される
# 各行には記事名が"title"キーに，記事本文が"text"キーの辞書オブジェクトに格納され，そのオブジェクトがJSON形式で書き出される
# ファイル全体はgzipで圧縮される
# 以下の処理を行うプログラムを作成せよ．


# %% 20. JSONデータの読み込み
# Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
# 問題21-29では，ここで抽出した記事本文に対して実行せよ
#
def get_wiki_UK_text():
    wikis = []
    with open('jawiki-country.json', encoding='utf-8') as f:
        for line in f:
            wikis.append(dict(json.loads(line)))
    uk = [wiki['text'] for wiki in wikis if 'イギリス' in wiki['text']]
    return uk


def ans_20():
    uk = get_wiki_UK_text()
    print(uk[0])


# %% 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ．
#
def ans_21():
    uk = get_wiki_UK_text()
    """
    num_article = len(wiki_UK)
    category_rows = []
    for idx_article in range(num_article):
        rows = wiki_UK[idx_article]["text"].split("\n")
        bool_of_category_name = \
            list(map(lambda x: "[[Category:" in x, rows))
        idx_of_category_name = \
            list(compress(list(range(len(rows))), bool_of_category_name))
        num_category_row = len(idx_of_category_name)
        category_rows.append([])
        for row_idx in range(num_category_row):
            category_rows[-1].append(rows[idx_of_category_name[row_idx]])
    """
    #
    # 正規表現を用いたスマートな解法
    # https://qiita.com/segavvy/items/73f1b91ff75529ae3b8d
    pattern = re.compile(r"""
                         ^(.*\[\[
                         Category
                         :.*\]\].*)$
                         """,
                         re.MULTILINE+re.VERBOSE)
    # re.compile 正規表現オブジェクトをコンパイルする
    # re.MULTILINE  行ごとに正規表現マッチする (デフォでは^や$は文章全体の最初と最後を見てしまう)
    # re.VERBOSE  複数行で正規表現を書けるフラグ
    # https://docs.python.jp/3/library/re.html#re.MULTILINE
    result = list(map(lambda x: pattern.findall(x), uk))
    # .findall  Return all non-overlapping matches of pattern in string, as a list of strings.
    for i in result:
        print(i)


# %% 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
def ans_22():
    uk = get_wiki_UK_text()
    pattern = re.compile(r'^\[\[Category:(.*?)(?:\|.*)?\]\].*$', re.MULTILINE)
    match = list(map(lambda x: pattern.findall(x), uk))
    print(match)
    # (.*?)　その1文字があってもなくてもよい
    # (?:)　カッコ内はキャプチャ対象外


# %% 23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば"== セクション名 =="なら1）を表示せよ．
def ans_23():
    uk = get_wiki_UK_text()
    pattern = re.compile(r'^=+.*=+$', re.MULTILINE)
    match = list(map(lambda x: pattern.findall(x), uk))
    ret = list(map(lambda x:
                list(map(lambda y:
                   y.replace('=', '').replace(' ', '') +
                   ':' +
                   f'{y.count("=")/2 - 1}',
                   x)), match))
    print(ret[0])
    # ans: ^(={2,})\s*(.+?)\s*\1.*$
    # (={2,}) # キャプチャ対象、2個以上の'='
    # \s*     # 余分な0個以上の空白（'哲学'や'婚姻'の前後に余分な空白があるので除去）
    # (.+?)   # キャプチャ対象、任意の文字が1文字以上、非貪欲（以降の条件の巻き込み防止）
    # \s*     # 余分な0個以上の空白
    # \1      # 後方参照、1番目のキャプチャ対象と同じ内容

# %% 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．
def ans_24():
    uk = get_wiki_UK_text()
    # pattern = re.compile(r'^\[\[(?:ファイル|File):(.*)(?:\|.*\|.*\|.*)$', re.MULTILINE)
    pattern = re.compile(r'^\[\[(?:ファイル|File):(.+?)\|', re.MULTILINE)
    match = list(map(lambda x: pattern.findall(x), uk))
    print(match[0])


# %% 25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
def ans_25():
    uks = get_wiki_UK_text()
    uk_basis = []
    reigai = {}
    for i, uk in enumerate(uks):
        basis = {}
        flg = False
        for row in uk.split('\n'):
            if row == '}}':
                break
            elif flg:
                try:
                    k, v = row[1:].split(' =')
                except:
                    reigai[i] = row
                basis[k] = v
            elif '{{基礎情報' in row:
                flg = True
            else:
                continue
        uk_basis.append(basis)

    print(uk_basis[0])

    return reigai

# %% 第4章: 形態素解析
"""
第4章: 形態素解析
夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，
その結果をneko.txt.mecabというファイルに保存せよ．
このファイルを用いて，以下の問に対応するプログラムを実装せよ．

なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．
"""


# %% 30. 形態素解析結果の読み込み
"""
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は
表層形（surface），
基本形（base），
品詞（pos），
品詞細分類1（pos1）
をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

chasenの出力フォーマットは以下
表層形\t読み\t基本形\t品詞(-品詞細分類1)(-品詞細分類2)(-品詞細分類3)(\t活用形\t活用型)
ex) 生れ	ウマレ	生れる	動詞-自立	一段	連用形
    {
    'surface': '生れ',
    'base': '生れる',
    'pos': '動詞',
    'pos1': '自立'
    }
1行ずつ辞書を入れてリストを作る
"""

def neko_wakati():
    neko_txt = []
    with open('neko.txt', encoding='utf-8') as f:
        neko_txt = f.read()
    m = MeCab.Tagger("-Owakati")
    neko = m.parse(neko_txt)

    with open('neko.txt.mecab', encoding='utf-8', mode='w') as f:
        f.write(neko)

neko_wakati()
def parse_neko():
    with open('neko.txt.mecab', encoding='utf-8', mode='r') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    for line in lines:
        line = line.strip('\n')
        if line == 'EOS':
            break
        results = line.split('\t')
        surf = results[0]
        base = results[2]
        pos = results[3].split('-')[0]
        if '-' in results[3]:
            pos1 = results[3].split('-')[1]
        else:
            pos1 = ''
        sentence.append({'surface': surf,
                         'base': base,
                         'pos': pos,
                         'pos1': pos1})
        if line.strip('\t')[0] == '。' or line.strip('\t')[0] == '」':
            sentences.append(sentence)
            sentence = []

    return sentences
"""
[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'},
{'surface': '\u3000', 'base': '\u3000', 'pos': '記号', 'pos1': '空白'},
{'surface': '吾輩', 'base': '吾輩', 'pos': '名詞', 'pos1': '代名詞'},
{'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'},
{'surface': '猫', 'base': '猫', 'pos': '名詞', 'pos1': '一般'},
{'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': ''},
{'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': ''},
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
"""


def ans_30():
    sentences = parse_neko()
    print(sentences[0])


# %% 31. 動詞
# 動詞の表層形をすべて抽出せよ．
def ans_31():
    sentences = parse_neko()
    verbs = []
    for sentence in sentences:
        for words in sentence:
            if words['pos'] == '動詞':
                verbs.append(words['surface'])
    print(verbs[:20])


# %% 32. 動詞の原形
# 動詞の原形をすべて抽出せよ．
def ans_32():
    sentences = parse_neko()
    verbs_base = []
    for sentence in sentences:
        for words in sentence:
            if words['pos'] == '動詞':
                verbs_base.append(words['base'])
    print(verbs_base[:20])


# %% 33. サ変名詞
# サ変接続の名詞をすべて抽出せよ．
def ans_33():
    sentences = parse_neko()
    norms_sahen = []
    for sentence in sentences:
        for words in sentence:
            if 'サ変' in words['pos1']:
                norms_sahen.append(words['base'])
    print(norms_sahen[:20])


# %% 34. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．
def ans_34():
    sentences = parse_neko()
    norm_ku = []
    for sentence in sentences:
        for i, _ in enumerate(sentence[:-2]):
            words = sentence[i:i+3]
            if words[0]['pos'] == '名詞' \
                and words[1]['surface'] == 'の' \
                and words[2]['pos'] == '名詞':
                norm_ku.append(''.join([x['surface'] for x in words]))
    print(norm_ku[:20])
ans_34()

# %% 35. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
def ans_35():
    sentences = parse_neko()
    # cont_norms = []
    ans = [[]]
    for sentence in sentences:
        cont_norm = []
        for words in sentence:
            if words['pos'] == '名詞':
                cont_norm.append(words['surface'])
            elif words['pos'] != '名詞':
                if len(cont_norm) > len(ans[0]):
                    ans = [cont_norm]
                elif len(cont_norm) == len(ans[0]):
                    ans.append(cont_norm)
                cont_norm = []
    print(ans)

ans_35()
"""
[['many', 'a', 'slip', "'", 'twixt', 'the', 'cup', 'and', 'the', 'lip'], ['明治', '三', '十', '八', '年', '何', '月', '何', '日', '戸締り']]
"""
# %% 36. 単語の出現頻度
# 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．
def calc_word_freq():
    """
      words  freq
    0     の  9194
    1     。  7486
    2     て  6873
    3     、  6772
    4     は  6422
    """
    sentences = parse_neko()

    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word['surface'])
    df_word_freq = pd.DataFrame(words, columns=['words'])
    df_word_freq = df_word_freq.groupby('words')\
                               .agg({'words': 'count'})\
                               .sort_values('words', ascending=False)
    df_word_freq = df_word_freq.rename(columns={'words': 'freq'})
    df_word_freq = df_word_freq.reset_index()

    return df_word_freq


def ans_36():
    df_word_freq = calc_word_freq()
    print(df_word_freq.iloc[:20, :])


# %% 37. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
def ans_37():
    df_word_freq = calc_word_freq()
    plt.bar(range(10), df_word_freq['freq'][:10])
    plt.xticks(range(10), df_word_freq['words'][:10])
    plt.title('Q.37')
    # plt.savefig('37.png')


# %% 38. ヒストグラム
# 単語の出現頻度のヒストグラム（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．
def ans_38():
    df_word_freq = calc_word_freq()
    df_word_freq_agg = df_word_freq[['freq']]\
                        .groupby('freq')\
                        .agg({'freq': 'count'})\
                        .rename(columns={'freq': 'count'})\
                        .reset_index()\
                        .sort_values('count', ascending=False)

    plt.bar(df_word_freq_agg.loc[:10, 'freq'],
            df_word_freq_agg.loc[:10, 'count'])
    plt.title('Q.38')

ans_38()

# %% 39. Zipfの法則
# 単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
def ans_39():
    df_word_freq = calc_word_freq()
    df_word_freq_agg = df_word_freq[['freq']]\
                        .groupby('freq')\
                        .agg({'freq': 'count'})\
                        .rename(columns={'freq': 'count'})\
                        .reset_index()\
                        .sort_values('count', ascending=False)
    plt.figure()
    plt.plot(1+np.arange(df_word_freq_agg.shape[0]),
             df_word_freq_agg['count'],
             'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Q.39')

ans_39()

# %% 40. 係り受け解析結果の読み込み（形態素）
# 形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）を
# メンバ変数に持つこととする．
# さらに，CaboChaの解析結果（neko.txt.cabocha）を読み込み，
# 各文をMorphオブジェクトのリストとして表現し，3文目の形態素列を表示せよ．
#
# xml形式で出力された結果を用いる　($ cabocha -f3 neko.txt > neko.txt.cabocha)
# -f1でもよいけどxmlの方がわかりやすそうなので
#  python標準ライブラリのxmlパーサーにも触れて経験値広がりそうでよい
# tok/feature = 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
# https://qiita.com/nezuq/items/f481f07fc0576b38e81d


def parse_neko_cabocha():
    with open('neko.txt.cabocha', encoding='utf-8') as f:
    # xml形式で出力された結果を用いる　($ cabocha -f3 neko.txt > neko.txt.cabocha)
        neko_cabocha = f.read()
    neko_cabocha = neko_cabocha.replace('<sentence>\n</sentence>\n', '')
    neko_cabocha = '<document>\n'+neko_cabocha+'</document>'
    # ネストにしないとコケる
    root = ET.fromstring(neko_cabocha)
    return root


class Morph():
    def __init__(self, surface, feature):
        feature = feature.split(',')
        self.surface = surface
        self.base = feature[-3]
        self.pos = feature[0]
        self.pos1 = feature[1]


def ans_40():
    root = parse_neko_cabocha()
    results = []
    sentence = []
    for i, row in enumerate(root.iter()):
        # xmlを行ごとに舐めるイテレーター
        if i == 0:
            continue
        if row.tag == 'tok':
            sentence.append(Morph(row.text, row.attrib['feature']))
            #
        elif row.tag == 'sentence':
            results.append(sentence)
            sentence = []
    results.append(sentence)
    for word in results[2]:
        print(f'surf: {word.surface}, ' +
              f'base: {word.base}, ' +
              f'pos:  {word.pos}, ' +
              f'pos1: {word.pos1}')

ans_40()


# %% 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），
# 係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，
# １文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．
class Chunk():
    def __init__(self, chunk):
        self.morphs = [Morph(word.text, word.attrib['feature'])
                        for word in chunk.iter() if word.tag=='tok']
        self.dst = self._get_dst(chunk)
        self.srcs = []

    def _get_dst(self, chunk):
        for ch in chunk.iter():
            # chunkの1iter目は文節の内容
            return int(ch.attrib['link'])

    def add_srcs(self, idx):
        self.srcs.append(idx)


def create_chunk_list():
    root = parse_neko_cabocha()
    chunks = []
    sentences = []
    for i, row in enumerate(root.iter()):
        if i == 0 or row.tag == 'tok':
            continue
        if row.tag == 'chunk':
            chunks.append(Chunk(row))
        elif row.tag == 'sentence':
            [chunks[chunk.dst].add_srcs(j) for j, chunk in enumerate(chunks)
             if chunk.dst != -1]
            sentences.append(chunks)
            chunks = []
    return sentences


def ans_41():
    sentences = create_chunk_list()
    [print(f'{"".join([word.surface for word in chunk.morphs])} dst:{chunk.dst}')
    for chunk in sentences[7]]

ans_41()

"""
しかも dst:8
あとで dst:2
聞くと dst:8
それは dst:8
書生という dst:5
人間中で dst:8
一番 dst:7
獰悪な dst:8
種族であったそうだ。 dst:-1
"""

# %% 42. 係り元と係り先の文節の表示
# 係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．
def ans_42():
    sentences = create_chunk_list()
    i = 0
    for chunks in sentences:
        for chunk in chunks:
            if chunk.dst != -1:
                src_words = "".join([morph.surface for morph in chunk.morphs])\
                    .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                dst_words = "".join([morph.surface for morph in chunks[chunk.dst].morphs])\
                    .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                print(f'{src_words}\t{dst_words}')
            i += 1
            if i >= 10:
                break
        if i >= 10:
            break

ans_42()

"""
吾輩は     猫である
名前は     無い
まだ      無い
　どこで    生れたか
生れたか    つかぬ
とんと     つかぬ
"""

# %% 43. 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．
def ans_43():
    sentences = create_chunk_list()
    i = 0
    for chunks in sentences:
        for chunk in chunks:
            if chunk.dst != -1:
                src_poses = [morph.pos for morph in chunk.morphs]
                dst_poses = [morph.pos for morph in chunks[chunk.dst].morphs]
                if ('名詞' in src_poses) and ('動詞' in dst_poses):
                    src_words = "".join([morph.surface for morph in chunk.morphs])\
                        .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                    dst_words = "".join([morph.surface for morph in chunks[chunk.dst].morphs])\
                        .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                    print(f'{src_words}\t{dst_words}')
                    i += 1
            if i >= 10:
                break
        if i >= 10:
            break

ans_43()

"""
　どこで    生れたか
見当が     つかぬ
所で      泣いて
ニャーニャー  泣いて
いた事だけは  記憶している
吾輩は     見た
ここで     始めて
ものを     見た
あとで     聞くと
我々を     捕えて
"""

# %% 44. 係り受け木の可視化
# 与えられた文の係り受け木を有向グラフとして可視化せよ．
# 可視化には，係り受け木をDOT言語に変換し，Graphvizを用いるとよい．
# また，Pythonから有向グラフを直接的に可視化するには，pydotを使うとよい．
def ans_44():
    sentences = create_chunk_list()
    for i, chunks in enumerate(sentences):
        if i <= 8:
            continue
        g = pydot.Dot(graph_type='digraph')
        for chunk in chunks:
            if chunk.dst != -1:
                src_words = "".join([morph.surface for morph in chunk.morphs])\
                    .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                src_node = pydot.Node(src_words)
                dst_words = "".join([morph.surface for morph in chunks[chunk.dst].morphs])\
                    .replace('。', '').replace('、', '').replace('「', '').replace('」', '')
                dst_node = pydot.Node(dst_words)

                g.add_node(src_node)
                g.add_node(dst_node)
                g.add_edge(pydot.Edge(src_node, dst_node))
        for node in g.get_node_list():
            node.set_fontname("IPAPGothic")

        FN_dot = f'neko_{i}.dot'
        FN_png = f'neko_{i}.png'
        with open(FN_dot, mode='w', encoding="utf-8") as f:
            f.write(g.to_string())
        subprocess.run(f'dot -T png {FN_dot} > {FN_png}', shell=True)
        break

ans_44()

# %% 45. 動詞の格パターンの抽出
"""
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
ただし，出力は以下の仕様を満たすようにせよ．

動詞を含む文節において，最左の動詞の基本形を述語とする
述語に係る助詞を格とする
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
「吾輩はここで始めて人間というものを見た」という例文（neko.txt.cabochaの8文目）を考える．
 この文は「始める」と「見る」の２つの動詞を含み，
 「始める」に係る文節は「ここで」，「見る」に係る文節は「吾輩は」と「ものを」と解析された場合は，
 次のような出力になるはずである．

始める  で
見る    は を
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

コーパス中で頻出する述語と格パターンの組み合わせ
「する」「見る」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
"""
def ans_45():
    sentences = create_chunk_list()
    # i = 0
    result = []
    for chunks in sentences:
        for chunk in chunks:
            poses = [morph.pos for morph in chunk.morphs]
            if '動詞' in poses:
                verb = [morph.base for morph in chunk.morphs
                        if morph.pos == '動詞'][0]
                for src in chunk.srcs:
                    for morph in chunks[src].morphs:
                        if morph.pos == '助詞':
                            verb += '\t'+morph.surface
                result.append(verb)
        # i += 1
        # if i >= 10:
        #     break
    result = '\n'.join(result)
    with open('neko_45.txt', mode='w', encoding="utf-8") as f:
        f.write(result)
    # print(result)

ans_45()

p = subprocess.run('cat neko_45.txt | sort | uniq -c | sort -nr'
               , stdout=subprocess.PIPE
               , shell=True)

"""
生れる     で
つく      か       が
する
泣く      で
する      て       だけ      は
始める     で
見る      は       を
聞く      で
捕える     を
煮る      て
食う      て
思う      から
"""
# $


# %%　46. 動詞の格フレーム情報の抽出
"""
45のプログラムを改変し，述語と格パターンに続けて
項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
「吾輩はここで始めて人間というものを見た」という例文（neko.txt.cabochaの8文目）を考える．
この文は「始める」と「見る」の２つの動詞を含み，
「始める」に係る文節は「ここで」，「見る」に係る文節は「吾輩は」と「ものを」と解析された場合は，
次のような出力になるはずである．

始める  で      ここで
見る    は を   吾輩は ものを
"""

def ans_46():
    sentences = create_chunk_list()
    i = 0
    result = []
    for chunks in sentences:
        for chunk in chunks:
            poses = [morph.pos for morph in chunk.morphs]
            if '動詞' in poses:
                verb = [morph.base for morph in chunk.morphs
                        if morph.pos == '動詞'][0]
                for src in chunk.srcs:
                    for morph in chunks[src].morphs:
                        if morph.pos == '助詞':
                            verb += '\t'+morph.surface
                for src in chunk.srcs:
                    verb += '\t'+''.join([morph.surface
                                          for morph in chunks[src].morphs])
                result.append(verb)
        i += 1
        if i >= 10:
            break
    result = '\n'.join(result)
    # with open('neko_46.txt', mode='w', encoding="utf-8") as f:
        # f.write(result)
    print(result)

ans_46()
"""
生れる     で       　どこで
つく      か       が       生れたか    とんと     見当が
する
泣く      で       所で      ニャーニャー
する      て       だけ      は       泣いて     いた事だけは
始める     で       ここで
見る      は       を       吾輩は     ものを
聞く      で       あとで
捕える     を       時々      我々を
煮る      て       捕えて
食う      て       煮て
思う      から      しかし     なかったから  恐し      いとも
"""

# %% 47. 機能動詞構文のマイニング
"""
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
述語は「サ変接続名詞+を+動詞の基本形」とし，
文節中に複数の動詞があるときは，最左の動詞を用いる
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
例えば「別段くるにも及ばんさと、主人は手紙に返事をする。」という文から，
以下の出力が得られるはずである．

返事をする      と に は        及ばんさと 手紙に 主人は
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

コーパス中で頻出する述語（サ変接続名詞+を+動詞）
コーパス中で頻出する述語と助詞パターン
"""
# 「及ばん　さ　と」　なのでは？　「さ」は削る？
def ans_47():
    sentences = create_chunk_list()
    result = []
    # flg = False
    for chunks in sentences:
        for i, chunk in enumerate(chunks):
            poses1 = [morph.pos1 for morph in chunk.morphs]
            surfaces = [morph.surface for morph in chunk.morphs]
            dst_poses = [morph.pos for morph in chunks[chunk.dst].morphs]
            # if '及ばんさ' in ''.join(surfaces):
            #     flg = True
            if 'サ変接続' in poses1 and\
                'を' in surfaces and\
                '動詞' in dst_poses:
                norm = ''.join([morph.surface for morph in chunk.morphs
                        if morph.pos1 == 'サ変接続'])
                verb = [morph.base for morph in chunks[chunk.dst].morphs
                        if morph.pos == '動詞'][0]

                curr = norm + 'を' + verb + '\t'

                for src in chunks[chunk.dst].srcs:
                    if src == i:
                        continue
                    for morph in chunks[src].morphs:
                        if morph.pos == '助詞':
                            curr += ' '+morph.surface
                curr += '\t'
                for src in chunks[chunk.dst].srcs:
                    if src == i:
                        continue
                    curr += ' '+''.join([morph.surface
                                     for morph in chunks[src].morphs])
                result.append(curr)
        # if flg:
        #     break
    result = '\n'.join(result)
    with open('neko_47.txt', mode='w', encoding="utf-8") as f:
        f.write(result)
    # print(result)

ans_47()


# %% 48. 名詞から根へのパスの抽出
"""
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
ただし，構文木上のパスは以下の仕様を満たすものとする．

各文節は（表層形の）形態素列で表現する
パスの開始文節から終了文節に至るまで，各文節の表現を"->"で連結する
「吾輩はここで始めて人間というものを見た」という文（neko.txt.cabochaの8文目）から，
次のような出力が得られるはずである．

吾輩は -> 見た
ここで -> 始めて -> 人間という -> ものを -> 見た
人間という -> ものを -> 見た
ものを -> 見た
"""


def get_upper_chunk(chunk, chunks, path_list):
    word = ''.join([morph.surface for morph in chunk.morphs])
    if chunk.dst == -1:
        path_list.append(word)
    else:
        get_upper_chunk(chunks[chunk.dst], chunks, path_list)
        path_list.append(word)


def ans_48():
    sentences = create_chunk_list()
    result = []
    i = 0
    for chunks in sentences:
        for chunk in chunks:
            path_list = []
            if '名詞' in [morph.pos for morph in chunk.morphs]:
                get_upper_chunk(chunk, chunks, path_list)
                if len(path_list) >= 1:
                    result.append(' -> '.join(path_list[::-1]))
        i += 1
        if i >= 7:
            break
    result = '\n'.join(result)
    print(result)


ans_48()

"""
吾輩は -> 猫である。
猫である。
名前は -> 無い。
　どこで -> 生れたか -> つかぬ。
見当が -> つかぬ。
何でも -> 薄暗い -> 所で -> 泣いて -> 記憶している。
所で -> 泣いて -> 記憶している。
ニャーニャー -> 泣いて -> 記憶している。
いた事だけは -> 記憶している。
記憶している。
吾輩は -> 見た。
ここで -> 始めて -> 人間という -> ものを -> 見た。
人間という -> ものを -> 見た。
ものを -> 見た。
"""

# %% 49. 名詞間の係り受けパスの抽出
"""
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

問題48と同様に，パスは開始文節から終了文節に至るまでの
各文節の表現（表層形の形態素列）を"->"で連結して表現する
文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
また，係り受けパスの形状は，以下の2通りが考えられる．

文節iから構文木の根に至る経路上に文節jが存在する場合:
    文節iから文節jのパスを表示
上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，
    文節kの内容を"|"で連結して表示
例えば，「吾輩はここで始めて人間というものを見た。」という文（neko.txt.cabochaの8文目）から，
次のような出力が得られるはずである．

Xは | Yで -> 始めて -> 人間という -> ものを | 見た
Xは | Yという -> ものを | 見た
Xは | Yを | 見た
Xで -> 始めて -> Y
Xで -> 始めて -> 人間という -> Y
Xという -> Y

（なぜ 「Xという | Yを | 見た」　がないのか？）
　→　「文節iから構文木の根に至る経路上に文節jが存在する場合」だから
（Xで -> 始めて -> Y）
（Xは | Yという -> ものを | 見た）
Yの付き方が違うのでは？

上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，
    文節kの内容を"|"で連結して表示

吾輩は | ここで -> 始めて -> 人間という -> ものを | 見た
吾輩は | 人間という -> ものを | 見た
吾輩は | ものを | 見た
ここで -> 始めて -> 人間という
ここで -> 始めて -> 人間という -> ものを
人間という -> ものを


上記以外で、「吾輩は」と「ここで」から構文木の根に至る経路上で共通の「見た」で交わる場合:
    「吾輩は」から「見た」に至る直前のパスと「ここで」から「見た」に至る直前までのパス，
    「見た」の内容を"|"で連結して表示

吾輩は | ここで -> 始めて -> 人間という -> ものを | 見た


1．名詞句は二つあるか？
　　2つある：パス上判定へ進む　違う：次
2．パス上か？
　　パス上：パス出力　違う：共通文節探索、パス出力

「名詞句」：動詞が入ってたら名詞句として扱わない

"""


def get_upper_chunk_pos(chunk, chunks, path_list):
    word = ''.join([morph.surface for morph in chunk.morphs])
    pos = [morph.pos for morph in chunk.morphs]
    if chunk.dst == -1:
        path_list.append({'word': word, 'pos': pos})
    else:
        get_upper_chunk(chunks[chunk.dst], chunks, path_list)
        path_list.append({'word': word, 'pos': pos})


def get_crossing(path_list_i, path_list_j):
    # 交差する文節、path_iにおけるidx, path_jにおけるidx
    # を返す
    for i, word_i in enumerate(path_list_i):
        if word_i in path_list_j:
            for j, word_j in enumerate(path_list_j):
                if word_j == word_i:
                    return word_i, i, j


def check_meisiku(chunk):
    # 「猫である」は名詞句ではない
    meisi_flg = np.any(['名詞' in morph.pos for morph in chunk.morphs])
    dousi_flg = np.any(['動詞' in morph.pos for morph in chunk.morphs])
    return meisi_flg and not dousi_flg


def get_XY(chunk, XY):
    XY_flg = False
    ret = ''
    for morph in chunk.morphs:
        if morph.pos == '名詞' and not XY_flg:
            ret += XY
            XY_flg = True
        else:
            ret += morph.surface
    return ret


def ans_49():
    sentences = create_chunk_list()
    result = []
    for k, chunks in enumerate(sentences):
        for i, chunk_i in enumerate(chunks[:-1]):

            # 残りの部分の名詞句flg
            meisiku_flg = [check_meisiku(rest_chunk) for rest_chunk in chunks[i:]]

            # 最後の名詞句である or 名詞句でない
            if np.sum(meisiku_flg) == 1 or not check_meisiku(chunk_i):
                continue

            # word_i = ''.join([morph.surface for morph in chunk_i.morphs])
            path_list_i = []
            get_upper_chunk(chunk_i, chunks, path_list_i)
            path_list_i = path_list_i[::-1]

            # 一つずつ
            for chunk_j in chunks[i+1:]:
                poses_j = [morph.pos for morph in chunk_j.morphs]
                if '名詞' not in poses_j:
                    continue
                word_j = ''.join([morph.surface for morph in chunk_j.morphs])
                if word_j in path_list_i:
                    # i -> ... -> j
                    # ret = word_i
                    ret = get_XY(chunk_i, 'X')
                    for x in path_list_i[1:]:
                        if x != word_j:
                            ret += ' -> ' + x
                        else:
                            ret += ' -> ' + get_XY(chunk_j, 'Y')
                            break
                else:
                    # i -> ... | j -> ... | k
                    path_list_j = []
                    get_upper_chunk(chunk_j, chunks, path_list_j)
                    path_list_j = path_list_j[::-1]
                    crossing, end_i, end_j = get_crossing(path_list_i, path_list_j)
                    ret = get_XY(chunk_i, 'X')
                    if len(path_list_i) > 1:
                        ret += ' -> '
                        ret += ' -> '.join(path_list_i[1:end_i])
                    if end_j != 0:
                        ret += ' | '
                        ret += get_XY(chunk_j, 'Y')
                        if len(path_list_j) >= 2:
                            ret += ' -> '
                            ret += ' -> '.join(path_list_j[1:end_j])
                        ret += ' | '
                    ret += crossing
                if ret == 'Yが | つかぬ。':
                    import pdb; pdb.set_trace()
                result.append(ret)
        if k >= 20:
            break
    result = '\n'.join(result)
    print(result)

ans_49()

"""
　Xで -> 生れたか | Yが ->  | つかぬ。
Xでも -> 薄暗い -> Yで
Xでも -> 薄暗い -> 所で | Y ->  | 泣いて
Xでも -> 薄暗い -> 所で -> 泣いて | Yだけは ->  | 記憶している。
Xでも -> 薄暗い -> 所で -> 泣いて -> Yしている。
Xで ->  | Y ->  | 泣いて
Xで -> 泣いて | Yだけは ->  | 記憶している。
Xで -> 泣いて -> Yしている。
X -> 泣いて | Yだけは ->  | 記憶している。
X -> 泣いて -> Yしている。
Xは ->  | Yで -> 始めて -> 人間という -> ものを | 見た。
Xは ->  | Yという -> ものを | 見た。
Xは ->  | Yを ->  | 見た。
Xで -> 始めて -> Yという
Xで -> 始めて -> 人間という -> Yを
Xという -> Yを
Xで -> 聞くと | Yは ->  | 種族であったそうだ。
"""

# %% main
# if __name__ == '__main__':
    # get_neko_dict()
    # ans_37()
    # exec("ans_" + str(q).zfill(2) + "()")
