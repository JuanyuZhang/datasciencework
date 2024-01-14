import jieba
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# from tkinter import _flatten

def trans(text):
    return text.strip("[]").replace("'", "").replace(",", " ")


s = ['不', '没', '说', '捂脸', '时', '月', '日', '还', '也', '怎么回事', '想', '都', '手机', '谢谢', '回复', '一个',
     "你好", "海港区", "秦皇岛", "青岛市", "山东省", "泰州市", "海陵区", "城东", "街道", "迎春", '好', '不了',
     '时间', '时间点', '感觉', '会', '能', '哭泣', 'Log', '捂', '脸', '抓', '很', '买', '后', '前', '江苏省', '泰州']


def remove_stopword(words):
    words = words.replace("  ",' ').split(' ')
    lis = [word for word in words if word not in s]
    return lis



tf_vectorizer = TfidfVectorizer()

data = pd.read_csv("低端缓存文件.csv")
data['描述'] = data['描述'].apply(trans).apply(remove_stopword)
wordlist = data['描述'].tolist()
wordlist = list(chain.from_iterable(wordlist))
wordfre = pd.Series(wordlist).value_counts()
wordfre = wordfre[:1010]
# print(wordfre)


def keepfre(li):
    li = [word for word in li if word in wordfre]
    return (' ').join(li)
#
data['描述'] = data['描述'].apply(keepfre)
# print(data['描述'])
X = tf_vectorizer.fit_transform(data.描述)
# X = tf_vectorizer.fit_transform(wordlist)
# print(X.toarray().sum(axis=0).tolist())
#
data1 = {'word': tf_vectorizer.get_feature_names_out(),
          'tfidf': X.toarray().sum(axis=0).tolist()}
df2 = pd.DataFrame(data1).sort_values(by="tfidf", ascending=False, ignore_index=True)
print(df2.head(35))


n_topics = 4  #分为4类
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                learning_method='batch',
                                learning_offset=100,
#                                 doc_topic_prior=0.1,
#                                 topic_word_prior=0.01,
                               random_state=0)
lda.fit(X)

def print_top_words(model, feature_names, n_top_words):
    tword = []
    tword2 = []
    tword3=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_pro=[str(round(topic[i],3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  #(round(topic[i],3))
        tword.append(topic_w)
        tword2.append(topic_pro)
        print(" ".join(topic_w))
        print(" ".join(topic_pro))
        print(' ')
        word_pro=dict(zip(topic_w,topic_pro))
        tword3.append(word_pro)
    return tword3

n_top_words = 10
feature_names = tf_vectorizer.get_feature_names_out()
word_pro = print_top_words(lda, feature_names, n_top_words)