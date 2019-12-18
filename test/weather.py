from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'http://www.weather.go.kr/weather/observation/currentweather.jsp'

response = requests.get(url)
# print(response)
soup = BeautifulSoup(response.text, 'html.parser')
# print(soup)
place_p = '#content_weather > table > tbody > tr > td:nth-child(1) > a'
places = [place.text for place in soup.select(place_p)]
#print(places)
temp_p = '#content_weather > table > tbody > tr > td:nth-child(6)'
temps = [temp.text for temp in soup.select(temp_p)]
#print(temps)
bodytemp_p='#content_weather > table > tbody > tr > td:nth-child(8)'
bodytemps = [bodytemp.text for bodytemp in soup.select(bodytemp_p)]
#print(bodytemps)
humid_p = '#content_weather > table > tbody > tr > td:nth-child(11)'
humids = [humid.text for humid in soup.select(humid_p)]
humids = [int(humid) / 100 for humid in humids]
#print(humids)
wind_p = '#content_weather > table > tbody > tr > td:nth-child(12)'
winds = [wind.text for wind in soup.select(wind_p)]
#print(winds)

frame = {'지점': places, '현재온도' : temps, '체감온도' : bodytemps, '습도' : humids, '풍향' : winds}
#print(pd.DataFrame(frame))
weather = pd.DataFrame(frame)
weather.to_csv('weather.csv', index=False)