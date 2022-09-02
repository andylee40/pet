#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:44:24 2022

@author: admin
"""
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC   
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from matplotlib.pyplot import imread
from snownlp import SnowNLP
import json
import requests
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import re
import emoji
import numpy as np




#過濾中文英文數字以外字符
def filter_str(desstr, restr=''):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)


#爬蟲程式
def scrape(counts):
    
    global df
    #儲存標題以利判斷
    data=[]
    #儲存所有資料
    data2=[]
    specount=0
    #指定瀏覽器開啟
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    #進入寵物版
    driver.get("https://www.dcard.tw/f/pet")

    #將視窗開至最大
    driver.maximize_window()
    #開始爬蟲
    while True:
        #等待加載特定元素出現
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME,'article')))
        #找出文章位置
        post=driver.find_elements(By.TAG_NAME,'article')
        try:
            for posts in post:
                #print(posts)
                push_list=[]
                #標題
                title=posts.find_element(By.TAG_NAME,'h2').text
                
                #喜歡數
                try:
                    like=int(posts.find_element(By.CSS_SELECTOR,'.sc-1b0f4fad-0.jizqFI').text)
                except:
                    like=0
                #回應數
                try:
                    respon=int(posts.find_element(By.CSS_SELECTOR,'.sc-9130b5d8-2.cYLwhJ').text)
                except:
                    respon=0
               
                #滾動特殊處理
                if specount==3:
                    height=posts.size['height']+192
                elif specount==4:
                    height=posts.size['height']+215
                else:
                    height=posts.size['height']
                
                #文章內容
                if title in data:
                    continue
                else:
                    print(title)
                    #文章連結
                    ahref=posts.find_element(By.TAG_NAME,'a')
                    #點擊文章連結
                    webdriver.ActionChains(driver).move_to_element(ahref ).click(ahref ).perform()
                    time.sleep(3)
                    #文章內文字
                    content=driver.find_element(By.CSS_SELECTOR,".sc-ba53eaa8-0.iSPQdL").text
                    
                    #關閉文章視窗
                    driver.find_element(By.CSS_SELECTOR,".sc-ab9e99c9-2.cEeHvv").click()
                    time.sleep(1)
                    #疊加
                    data.append(title)
                    push_list.append(filter_str(title))
                    push_list.append(like)
                    push_list.append(respon)
                    push_list.append(filter_str(content))
                    #push_list.append(filter_str(title)+':'+filter_str(content))
                    #with open('explore.txt', 'a', encoding='utf-8') as file:
                        #file.write(filter_str(title)+':'+filter_str(content)+'\n')
                    #push_list.append(filter_str(content))
                #抓取熱門前100篇文章
                if len(data) > counts:
                     print('工作完成')
                     break   
                #向下滾動到下一個元素出現
                driver.execute_script("window.scrollBy(0,"+str(height)+");")
                time.sleep(1)
                #data2.append(push_list[0])
                data2.append(push_list)
                specount=specount+1
            df=pd.DataFrame(data=data2,columns=['title','like','response','content'])
        except:
            continue
        #抓取熱門前100篇文章
        if len(data) > counts:
             break
        
    #關閉瀏覽器
    driver.quit()

if __name__=='__main__':
    page=40
    scrape(page)
    print(df.corr())


#str_array = []
#with open('explore.txt','r') as f:
    #for line in f.readlines():
        #str_array.append(line)
        
#f.close()        


# #文字雲製作
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# #深度學習
# #斷詞
# ws = WS("../ptt關鍵字/data", disable_cuda=False)
# #詞性標記
# pos = POS("../ptt關鍵字/data", disable_cuda=False)
# #實體辨識
# ner = NER("../ptt關鍵字/data", disable_cuda=False)
# word_sentence_list = ws(
#     data[1:],
#     sentence_segmentation=True, # To consider delimiters
#     segment_delimiter_set = {",", "。", ":", "?", "!", ";", ".", "（", "）", "", "()", " [",
#         "] ", ":", "", "》"}, 
# )
# pos_sentence_list = pos(word_sentence_list)
# entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

# #釋放記憶體
# del ws
# del pos
# del ner

# #顯示結果
# def print_word_pos_sentence(word_sentence, pos_sentence):
#     assert len(word_sentence) == len(pos_sentence)
#     for word, pos in zip(word_sentence, pos_sentence):
#         print(f"{word}({pos})", end="\u3000")
#     print()
#     return
    
# for i, sentence in enumerate(word_sentence_list):
#     print()
#     print(f"'{sentence}'")
#     print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
#     for entity in sorted(entity_sentence_list[i]):
#         print(entity)

# #統計字詞數量
# count_list = []
# for e in entity_sentence_list:
#     for i in e:
#         count_list.append(i[3])        
# df = pd.DataFrame(count_list, columns=["entity"])
# text = df.entity.value_counts()
# text.head(30)        


# #引用字體
# text2 = " ".join(review for review in count_list)
# font_path = '/Users/admin/Downloads/NotoSansCJK-Regular.ttc'

# #color_mask = imread('dog.jpg')
# #cloud = WordCloud(font_path=font_path,
# #background_color="black",
# #mask=color_mask,
# #max_words=2000,
# #max_font_size=80,
# #random_state=42,
# #relative_scaling=0)
# #word_cloud = cloud.generate(text2)
# #plt.axis('off')
# #plt.imshow(word_cloud)
# #plt.show()


# #文字雲圖片引入與調整
# mask_color = np.array(Image.open('../ptt關鍵字/parrot-by-jose-mari-gimenez2.jpg'))
# mask_color = mask_color[::3, ::3]
# mask_image = mask_color.copy()
# mask_image[mask_image.sum(axis=2) == 0] = 255
# edges = np.mean([gaussian_gradient_magnitude(mask_color[:, :, i]/255., 2) for i in range(3)], axis=0)
# mask_image[edges > .08] = 255
# #繪製文字雲
# wordcloud = WordCloud(width=1200, height=600,max_font_size=100, max_words=2000, random_state=42,mask=mask_image,relative_scaling=0,font_path=font_path).generate(text2)
# image_colors = ImageColorGenerator(mask_color)
# wordcloud.recolor(color_func=image_colors)
# plt.figure()
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()
# plt.savefig('data.png', dpi = 600)

      
        
# #關聯性分析
# fq_count_lists = []
# for e in entity_sentence_list:
#     fq_count_list = []

#     for i in e:
#         fq_count_list.append(i[3])
#     fq_count_lists.append(fq_count_list)
# te = TransactionEncoder()
# te_ary = te.fit(fq_count_lists).transform(fq_count_lists)
# df = pd.DataFrame(te_ary, columns=te.columns_)  
# fpgrowth(df, min_support=0.1, use_colnames=True)        


# for j in word_sentence_list[2]:
#     s=SnowNLP(j)
#     if s.sentiments <=0.3:
#         continue
#         print(j,str(s.sentiments)+'\n')
#     else:
#         print(j,str(s.sentiments)+'\n')   