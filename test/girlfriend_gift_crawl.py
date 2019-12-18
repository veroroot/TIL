import requests
import urllib.request
import cx_Oracle
import os
import time
import re


os.putenv('NLS_LANG', '.UTF8')
connection = cx_Oracle.connect('hr', '1234', 'localhost/xe')
cursor = connection.cursor()

client_id = 'mxwcqQBBrs_ggTLo7Ihy'
client_secret = 'wMuOBLfHqK'

display = 20
start = 1

headers = {"X-Naver-Client-Id" : client_id, "X-Naver-Client-Secret" : client_secret}

keyword = urllib.parse.quote("여자친구 선물")

url = "https://openapi.naver.com/v1/search/blog?query=" + keyword + "&display="+ str(display) + "&start="+str(start)
response = requests.get(url, headers=headers)
information = response.json()
total = information['total']
#print(total)
total_page = 0
if total%20 == 0:
    total_page = total//20
else:
    total_page = total//20+1

def get_api_result(keyword, display, start):
    url = "https://openapi.naver.com/v1/search/blog?query=" + keyword + "&display="+ str(display) + "&start="+str(start)
    response = requests.get(url, headers=headers)
    return response.json()

def call_and_print(keyword, page):
    json_obj = get_api_result(keyword, 20, (page-1)*20+1)
    #print(json_obj)
    for item in json_obj["items"]:
        title = remove_tag(item['title']).replace("'", "&quot;")
        description = remove_tag(item['description']).replace("'", "&quot;")
        bloggername = item['bloggername'].replace("'", "&quot;")
        bloggerlink = item['bloggerlink']
        print("title:",title,"description:",description,"bloggername:",bloggername,"bloggerlink:",bloggerlink)
        insert_sql = f"insert into blog_info values(blog_seq.nextval, '{title}', '{description}', '{bloggername}', '{bloggerlink}')"
        print("insert_sql:", insert_sql)
        cursor.execute(insert_sql)
        connection.commit()

def remove_tag(content):
    cleanr =re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', content)
    return cleantext

for page in range(1, total_page):
    print(page, "page")
    call_and_print(keyword, page)
    time.sleep(0.5)
