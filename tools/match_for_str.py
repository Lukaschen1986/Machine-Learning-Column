# -*- coding: utf-8 -*-
"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pypinyin
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FuzzyWuzzy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-Levenshtein
"""
from pypinyin import (pinyin, lazy_pinyin)
from fuzzywuzzy import (fuzz, process)
from scipy.spatial import distance
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
import numpy as np
import pandas as pd

'''
hotelName: hotelId
hotelPinYin: hotelId
hotelPinYin: hotelName

干扰词过滤
全部小写
'''
# offline
lst_hotelName = [
    "深圳安朴悦庭",
    "深圳安朴泊莱",
    "深圳滨河时代亚朵S酒店",
    "H酒店深圳宝安国际机场店",
    "秋果S酒店深圳福田会展中心店",
    "深圳海德酒店",
    "新桃园酒店深圳海岸店",
    "深圳T酒店公寓"
    ]

lst_hotelName_pinyin = [" ".join(lazy_pinyin(name)) for name in lst_hotelName]
tv_city = TfidfVectorizer(analyzer="word", stop_words=None, use_idf=True, smooth_idf=True)
array_hotelName_vec = tv_city.fit_transform(lst_hotelName_pinyin).toarray()
# tv_city.get_feature_names()

# online
query = "深圳新桃园"
query_pinyin = " ".join(lazy_pinyin(query))

array_query_vec = tv_city.transform([query_pinyin]).toarray()
array_similarity = 1 - distance.cdist(array_query_vec, array_hotelName_vec, metric="cosine")
sery_similarity = pd.Series(array_similarity[0])
sery_similarity = sery_similarity.sort_values(ascending=False)
sery_similarity_top = sery_similarity[0:3]
lst_idx = sery_similarity_top.index.tolist()
lst_hotelName_pinyin_top = [lst_hotelName_pinyin[idx] for idx in lst_idx]

hotelPinYin, score = process.extractOne(query_pinyin, lst_hotelName_pinyin_top, scorer=fuzz.WRatio)



