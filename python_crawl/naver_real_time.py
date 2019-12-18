import requests
from bs4 import BeautifulSoup

url = 'http://www.naver.com'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
#PM_ID_ct > div.header > div.section_navbar > div.area_hotkeyword.PM_CL_realtimeKeyword_base > div.ah_roll.PM_CL_realtimeKeyword_rolling_base > div > ul > li:nth-child(1) > a > span.ah_k
crawls = soup.select('#PM_ID_ct > div.header > div.section_navbar > div.area_hotkeyword.PM_CL_realtimeKeyword_base > div.ah_roll.PM_CL_realtimeKeyword_rolling_base > div > ul > li > a > span.ah_k')
#print(crawls)
crawl_list = [crawl.text for crawl in crawls]
print(crawl_list)
