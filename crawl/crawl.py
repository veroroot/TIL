from bs4 import BeautifulSoup
import requests
import random
import time

url = 'https://www.siksinhot.com/taste?upHpAreaId=9&hpAreaId=1122'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
#tabMove1 > div > ul > li:nth-child(1) > div > a > div > div > strong
key_selector = '#tabMove1 > div > ul > li > div > a > div > div > strong'
keys = soup.select(key_selector)
#print(keys)
key_list = [key.text for key in keys]
#print(key_list)
#tabMove1 > div > ul > li:nth-child(1) > div > a > span > img
img_selector = '#tabMove1 > div > ul > li > div > a > span > img'
imgs = soup.select(img_selector)
#print(imgs)
img_list = [img.get('src') for img in imgs]
#print(img_list)
menu=dict(zip(key_list, img_list))
print(menu)
menu_select = random.choice(menu.keys)