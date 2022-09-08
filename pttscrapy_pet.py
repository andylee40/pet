#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:30:59 2022

@author: admin
"""
import random
import datetime
import pandas as pd
import requests
import re
import time
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from matplotlib.pyplot import imread
#goal:爬取過去6個月的資料
today=datetime.date.today()
enddate=today-datetime.timedelta(days=30)


#爬蟲 
i=0
data=[]
#首頁網址
hrefs=['/bbs/pet/index.html']
while True:
    time.sleep(random.randint(1,3))
    url='https://www.ptt.cc'+hrefs[i]
    print('第'+str(i+1)+'頁')
    source=requests.get(url)
    content=source.text
    tree=BeautifulSoup(content,'lxml')
    row=tree.find_all('div',class_='r-ent')
    href=tree.find('a',string='‹ 上頁').get('href')
    hrefs.append(href)
    for rows in reversed(row):
            day=rows.find(class_='date').text.strip()
            checkday=datetime.datetime.strptime('2022/'+day,"%Y/%m/%d").date()
            if i !=0 and checkday < enddate:
                stop=1
                break
            else:
                #標題
                title=re.sub(r"\s+", "",rows.find(class_='title').text)
                #data.append(title)
                #print(checkday)
                
                #內文
                if type(rows.find(class_='title').find('a'))!=type(None):
                    push_list=[]
                    #標題
                    push_list.append(title)
                    
                    inside_href='https://www.ptt.cc'+rows.find(class_='title').find('a').get('href')
                    inside_source=requests.get(inside_href)
                    inside_content=inside_source.text
                    inside_tree=BeautifulSoup(inside_content,'lxml')
                    
                    #內文
                    inside_text=inside_tree.find('div',id='main-content')
                    all_text=inside_text.text
                    main_text=all_text.split('--')[0]
                    main=main_text.split('\n')
                    mainn=main[1:]
                    mainnn='\n'.join(mainn)
                    push_list.append(mainnn)
                   
                    #留言
                    for push in inside_tree.find_all("span", class_="push-content"):
                        push_list.append(push.text[2:])
                    data.append(push_list)
                else:
                   pass
    if i !=0 and checkday < enddate:
        break
    else:
        i=i+1

#文字雲製作
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#深度學習
#斷詞
ws = WS("./data", disable_cuda=False)
#詞性標記
pos = POS("./data", disable_cuda=False)
#實體辨識
ner = NER("./data", disable_cuda=False)
word_sentence_list = ws(
    data,
    sentence_segmentation=True, # To consider delimiters
    segment_delimiter_set = {",", "。", ":", "?", "!", ";", ".", "（", "）", "", "()", " [",
        "] ", ":", "", "》"}, 
)
pos_sentence_list = pos(word_sentence_list)
entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

del ws
del pos
del ner

def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return
    
for i, sentence in enumerate(word_sentence_list):
    print()
    print(f"'{sentence}'")
    print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
    for entity in sorted(entity_sentence_list[i]):
        print(entity)

#統計字詞數量
count_list = []
for e in entity_sentence_list:
    for i in e:
        count_list.append(i[3])        
df = pd.DataFrame(count_list, columns=["entity"])
text = df.entity.value_counts()
text.head(30)        


text2 = " ".join(review for review in count_list)
font_path = '/Users/admin/Downloads/NotoSansCJK-Regular.ttc'



#color_mask = imread('dog.jpg')
#cloud = WordCloud(font_path=font_path,
#background_color="black",
#mask=color_mask,
#max_words=2000,
#max_font_size=80,
#random_state=42,
#relative_scaling=0)
#word_cloud = cloud.generate(text2)
#plt.axis('off')
#plt.imshow(word_cloud)
#plt.show()


#文字雲圖片引入與調整
mask_color = np.array(Image.open('parrot-by-jose-mari-gimenez2.jpg'))
mask_color = mask_color[::3, ::3]
mask_image = mask_color.copy()
mask_image[mask_image.sum(axis=2) == 0] = 255
edges = np.mean([gaussian_gradient_magnitude(mask_color[:, :, i]/255., 2) for i in range(3)], axis=0)
mask_image[edges > .08] = 255
#繪製文字雲
wordcloud = WordCloud(width=1200, height=600,max_font_size=100, max_words=2000, random_state=42,mask=mask_image,relative_scaling=0,font_path=font_path).generate(text2)
image_colors = ImageColorGenerator(mask_color)
wordcloud.recolor(color_func=image_colors)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.savefig('data.png', dpi = 600)

      
        
#關聯性分析
fq_count_lists = []
for e in entity_sentence_list:
    fq_count_list = []
    for i in e:
        fq_count_list.append(i[3])
    fq_count_lists.append(fq_count_list)
te = TransactionEncoder()
te_ary = te.fit(fq_count_lists).transform(fq_count_lists)
df = pd.DataFrame(te_ary, columns=te.columns_)  
fpgrowth(df, min_support=0.02, use_colnames=True)        
      
        
      
        
      