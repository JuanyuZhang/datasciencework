import jieba
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from itertools import chain
from collections import Counter



file = "comment-high.csv"
picfile = "高端类别前8饼图.png"
data = pd.read_csv(file)
data = data.dropna(axis=0, how='any')
data = data.reset_index(drop=True)
pic = data['类别'].value_counts()
pic = pic.head(8)
print(pic.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
colors = ["#4CAF50","red","hotpink","#556B2F","#d5695d", "#5d8ca8", "#65a479", "#a564c9"]
plt.pie(pic,labels=pic.index,  colors = colors, autopct='%.2f%%',
        pctdistance=0.9, explode=(0, 0, 0, 0, 0, 0, 0.4, 0.2), shadow=True)
plt.title(picfile)
plt.savefig(picfile)
plt.show()
