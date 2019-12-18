import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/"

# 코스피 정보 가져오기

# 해당 url로 요청보내기
response = requests.get(url)
# bs4를 이용하여 파싱
soup = BeautifulSoup(response.text, 'html.parser') # 분석하기 쉽게 파싱을 한다.
# 원하는 데이터 뽑아오기
KOSPI_now = soup.find(id='KOSPI_now').text
#kospi = soup.select_one('#KOSPI_now')

print(KOSPI_now)