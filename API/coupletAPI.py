# coding:utf-8
import requests
import json
import base64
import imgAPI

class Couplet():
    def __init__(self,center,first,second):
        self.center=center
        self.first=first
        self.second=second

    def printinfo(self):
        print("横批：",self.center)
        print("上联：",self.first)
        print("下联：",self.second)


def get_access_token():

	API_Key = 'MI2Ugjh97pedpexYxkU8ia8F'
	Secret_Key = 'fgoVGaeiANDwkXhVxSiO8nrCCFxf5G6s'
	r = requests.post('https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+API_Key+'&client_secret='+Secret_Key)
	#print(r.text)
	j = json.loads(r.text)
	access_token = j.get('access_token')
	print(access_token )
	return access_token


body = {
    'text': '',
    'index': 0
}
headers = {
    'Content-Type': 'application/json',
}
token = get_access_token()  # 我的token参数
couplets=[]

# function: 获取对联
def coupletsGet(keyword):
    body['text'] = keyword
    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/couplets' + '?access_token=' + token
    param = json.dumps(body).encode('utf-8')
    result = requests.post(url=url, headers=headers, data=param)
    try:
        result = requests.post(url=url, headers=headers, data=param)
        center=result.json()['couplets']['center']
        first=result.json()['couplets']['first']
        second=result.json()['couplets']['second']
        temp=Couplet(center,first,second)
        couplets.append(temp)
    except:
        print('暂时没有找到')

keyword=imgAPI.getKeyword()

for i in keyword:
    #print(i)
    coupletsGet(i)
def getCouplets():
   return couplets
