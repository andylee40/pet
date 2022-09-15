#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:05:37 2022

@author: admin
"""
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude

#讀檔
df=pd.read_csv('/Users/admin/Desktop/pet2/ptt關鍵字/ptt_20220913_scrapy.csv',dtype='object',keep_default_na=False)

#將標題與內文合再一起
df['all']=df['title']+df['content']
df_2019=df[df['year']=='2019']
df_2020=df[df['year']=='2020']
df_2021=df[df['year']=='2021']
df_2022=df[df['year']=='2022']


#停用詞設置
stopwords = {}.fromkeys([line.strip() for line in open('/Users/admin/Desktop/pet2/ptt關鍵字/stopwords.txt', 'r', encoding='utf-8').readlines()])

#合併文字
d=''.join(df_2019['all'])
#Sentence = jieba.cut_for_search(d) 
Sentence = jieba.cut(d)
#print('/'.join(Sentence))
text = ' '.join(Sentence)

#詞頻
freq=WordCloud().process_text(text)
df_freq=pd.DataFrame(data=list(freq.items()),columns=['單字','次數']).sort_values('次數',ascending=False)

#關鍵字提取
keyword=jieba.analyse.extract_tags(text, topK = 10, withWeight = True, allowPOS = ())
df_key=pd.DataFrame(data=keyword,columns=['單字','權重'])


#文字雲圖片
mask_color = np.array(Image.open('/Users/admin/Desktop/pet2/ptt關鍵字/parrot-by-jose-mari-gimenez2.jpg'))
mask_color = mask_color[::3, ::3]
mask_image = mask_color.copy()
mask_image[mask_image.sum(axis=2) == 0] = 255
edges = np.mean([gaussian_gradient_magnitude(mask_color[:, :, i]/255., 2) for i in range(3)], axis=0)
mask_image[edges > .08] = 255

# 文字雲樣式設定
wc = WordCloud(font_path='/Users/admin/Downloads/NotoSansCJK-Regular.ttc', #設置字體
               max_words = 200,      #文字雲顯示最大詞數
               stopwords=stopwords,  #停用字詞
               mask=mask_image).generate_from_text(text)     
image_colors = ImageColorGenerator(mask_color)
wc.recolor(color_func=image_colors)
plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.title('ptt_2019')
plt.show()
plt.savefig('ptt_2019_v2.png',dpi=1000)
