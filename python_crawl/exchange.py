import requests
from bs4 import BeautifulSoup

url = 'https://finance.naver.com/marketindex/?tabSel=exchange#tab_section'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
exchange = soup.select_one('#exchangeList > li.on > a.head.usd > div > span.value').text

#exchange = soup.find(id = 'exchangeList').find('div', {"class":'head_info point_up'}).find('span', {'class':'value'}).text
print(exchange)
