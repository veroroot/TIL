import requests
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId
from gridfs import GridFSBucket
import urllib.request

# image url crawlling

client_id = 'mxwcqQBBrs_ggTLo7Ihy'
client_secret = 'wMuOBLfHqK'

headers = {"X-Naver-Client-Id" : client_id, "X-Naver-Client-Secret" : client_secret}

def get_api_result(keyword, display, start):
    url = "https://openapi.naver.com/v1/search/image?query=" + keyword + "&display="+ str(display) + "&start="+str(start)
    response = requests.get(url, headers=headers)
    return response.json()

def call_and_print(keyword, total_page):
    image_url_list = []
    for page in range(1,total_page):
        json_obj = get_api_result(keyword, 20, (page-1)*20+1)
        print(json_obj)
        for item in json_obj['items']:
            image_url_list.append(item['link'])
    return image_url_list

keyword = '박보검'
image_park = call_and_print(keyword, 50)

# image to MongoDB

db = MongoClient().python_test
fs = GridFS(db)
bucket = GridFSBucket(db)

for url in image_park:
    try:
        image = urllib.request.urlopen(url).read()
        image_type = url.split(".")[-1]
        content_type = f"image/{image_type}"
        image_name=url.split("/")[-1]
        grid_in = bucket.open_upload_stream(image_name, metadata={"contentType":content_type, "type":"human"})
        grid_in.write(image)
        grid_in.close()
        print("image_type:", image_type, ":content_type:",content_type,":image_name:",image_name)
    except:
        print("에러 발생")