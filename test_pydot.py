# -*- coding: utf-8 -*-
"""
pydotをためす
pydotで日本語を表示する

日本語表示の方法
・dotファイルのノードに一つずつ記入する
　→　dot形式にイチイチ出力するのは面倒
・pydotでノードを一つずつ設定する
　→　手間だけど安全
・pydotそのものを書き換える
　→　不安
・pydotでdotファイル作成＋書き込み、graphvizで描画
　→　pydotで修正が上手くいかないので試す
 　https://www.kunihikokaneko.com/dblab/toolchain/graphviz.html
  できた
  　・binにpath
   ・$ dot -T png test.dot > out.png
   
ポイント
 nodeごとにfontを設定する
　dot形式で出力してcmdで蹴る（ここはsubprocessにしたい)   
"""
import subprocess
import pydot
graph = pydot.Dot(graph_type='digraph')

node_a = pydot.Node("我が輩は")
node_b = pydot.Node("猫である")
node_c = pydot.Node("名前は")
node_d = pydot.Node("まだない")

graph.add_node(node_a)
graph.add_node(node_b)
graph.add_node(node_c)
graph.add_node(node_d)

graph.add_edge(pydot.Edge(node_a, node_b))
graph.add_edge(pydot.Edge(node_b, node_c))
graph.add_edge(pydot.Edge(node_c, node_d))
graph.add_edge(pydot.Edge(node_d, node_a))

for node in graph.get_node_list():
    node.set_fontname("IPAPGothic")

FN_dot = 'test.dot'
FN_png = 'test.png'
with open('test.dot', mode='w', encoding="utf-8") as f:
    f.write(graph.to_string())

subprocess.run(f'dot -T png {FN_dot} > {FN_png}', shell=True)
