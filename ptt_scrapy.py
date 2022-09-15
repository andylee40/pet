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
import seaborn as sns 
import matplotlib.pyplot as plt 
from bs4 import BeautifulSoup
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from matplotlib.pyplot import imread

#goal:爬取過去6個月的資料
# today=datetime.date.today()
# enddate=today-datetime.timedelta(days=730)
enddate=datetime.date.fromisoformat('2018-12-31')


#過濾中文英文數字以外字符
def filter_str(desstr, restr=''):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)


#爬蟲 
i=0
data=[]
#data2=[]
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
            # day=rows.find(class_='date').text.strip()
            # checkday=datetime.datetime.strptime('2022/'+day,"%Y/%m/%d").date()
        
            #標題
            title=re.sub(r"\s+", "",rows.find(class_='title').text)
            #data.append(title)
            #print(checkday)
                
            #內文
            if type(rows.find(class_='title').find('a'))!=type(None):
                push_list=[]
                #標題
                #push_list.append(title)
                #data2.append(title)
                inside_href='https://www.ptt.cc'+rows.find(class_='title').find('a').get('href')
                inside_source=requests.get(inside_href)
                inside_content=inside_source.text
                inside_tree=BeautifulSoup(inside_content,'lxml')
                
                    
                #內文時間
                try:
                    inside_time=inside_tree.find_all('span',class_='article-meta-value')[3].text
                    checktime=datetime.datetime.strptime(datetime.datetime.strptime(inside_time,"%a %b %d %H:%M:%S %Y").strftime("%Y/%m/%d"),"%Y/%m/%d").date()
                except:
                    pass
                #設定時間條件
                if i !=0 and checktime < enddate:
                    break
                
                print(checktime)
                #內文
                try:
                    inside_text=inside_tree.find('div',id='main-content')
                    all_text=inside_text.text
                    main_text=all_text.split('--')[0]
                    main=main_text.split('\n')
                    mainn=main[1:]
                    mainnn='\n'.join(mainn)
                except:
                    pass
                #data2.append(mainnn)
                
                push_list.append(checktime)
                push_list.append(checktime.strftime("%Y"))
                push_list.append(checktime.strftime("%m"))
                push_list.append(filter_str(title))
                push_list.append(filter_str(mainnn))
                data.append(push_list)
                   
                #留言
                #for push in inside_tree.find_all("span", class_="push-content"):
                    #push_list.append(push.text[2:])
                    #data2.append(push.text[2:])
                #data.append(push_list)
            else:
                  pass
    #設定時間條件
    if i !=0 and checktime < enddate:
        break
    else:
        i=i+1
        

#資料儲存
df=pd.DataFrame(data=data,columns=['date','year','month','title','content'])
df=df[df['date']>= enddate]
df.to_csv('ptt_'+(datetime.datetime.now().strftime("%Y%m%d"))+'_scrapy_1.csv',index=False)
#print(df['month'].value_counts())

#釋放記憶體
del data

#依年份整理文章數
statistics=pd.pivot_table(df,index='month',columns='year',values='content',aggfunc=['count'])
statistics.columns = statistics.columns.droplevel(0) 
statistics.columns.name = None               
statistics = statistics.reset_index()        

#敘述統計
print(statistics.describe())
print(statistics.sum(axis=0))

#依月分畫圖
plt.plot(statistics['month'],statistics['2019'],marker='^',label='2019')
plt.plot(statistics['month'],statistics['2020'],marker='o',label='2020')
plt.plot(statistics['month'],statistics['2021'],marker='*',label='2021')
plt.plot(statistics['month'],statistics['2022'],marker='s',label='2022')
plt.legend() 
plt.title('content count between 2019 and 2021')
plt.xlabel('month')
plt.ylabel('count')
plt.show()
plt.savefig('ptt_change.png')



      
        
      
        
      