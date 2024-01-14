# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import jieba
import pandas as pd
import re
import numpy as np
from itertools import chain
from collections import Counter


# Press the green button in the gutter to run the script.
pattern = r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_^{|}~—！，。？、￥…（）：【】《》‘’“”\s]+"
re_obj = re.compile(pattern)


def clear(text):
    if pd.isnull(text):
        print(type(text))
        return np.nan
    return re.sub(pattern, "", text)


def cut_word(text):  # 返回生成器
    return jieba.cut(text)


def get_stopword():  # 使用set
    s = set()
    with open('stopwords_baidu.txt', encoding='UTF-8') as f:
        for line in f:
            s.add(line.strip())
    return s


def remove_stopword(words):
    lis = [word for word in words if word not in stopword]
    # return lis
    return " ".join(lis)

file = "comment-mid.csv"
data = pd.read_csv(file)
data = data.dropna(axis=0, how='any')

data['描述'] = data['描述'].apply(clear)
# 2.分词 用jieba来实现分词
data['描述'] = data['描述'].apply(cut_word)

# 3.删掉停用词
stopword = get_stopword()
data['描述'] = data['描述'].apply(remove_stopword)
print(data.描述)
data.drop(['机型', 'IMEI', '日期'], axis=1).to_csv("中端缓存文件.csv")

# 4.词汇统计
li_2d = data['描述'].tolist()
li_1d = list(chain.from_iterable(li_2d))
print(f'总词汇量：{len(li_1d)}')
c = Counter(li_1d)
print(f'不重复词汇量：{len(c)}')
print(data.描述)