import matplotlib.pyplot as plt
import wordcloud
import numpy as np

from itertools import chain
from collections import Counter
import pandas as pd

file = "高端缓存文件.csv"
data = pd.read_csv(file)

def trans(text):
    return text.strip("[]").replace("'","").replace(",", " ")


data['描述'] = data['描述'].apply(trans)

text2d = data['描述'].tolist()

text1d = ''.join((text2d))
# print(text1d)
s = ['不', '没', '说', '捂脸', '时', '月', '日','还','也','怎么回事','想','都','手机','谢谢', '回复', '一个', "你好",
     '时间', '时间点','感觉','会','能','哭泣','Log','捂', '脸','抓','很','买','后','前']
print(text1d)
wc = wordcloud.WordCloud(font_path="C:\\Windows\\Fonts\\simsun.ttc",
                         width = 1000,
                         height = 700,
                         background_color='white',
                         max_words=70,stopwords=s)

wc.generate(text1d)
wc.to_file('高端-词云.png')