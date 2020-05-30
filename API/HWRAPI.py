#-*-coding:utf-8-*-
import requests
import json
import base64


def get_file_content(filePath):
	""" 读取图片base64 """
	with open(filePath, 'rb') as fp:
		return base64.b64encode(fp.read())


def get_access_token():

	API_Key = 'mT4a3WRy7Oiyua5Xe48qbk6T'
	Secret_Key = 'NVjhjGMf1ktLTMhiuCuGIGIOD6AvgjrZ'
	r = requests.post('https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+API_Key+'&client_secret='+Secret_Key)
	#print(r.text)
	j = json.loads(r.text)
	access_token = j.get('access_token')
	print(access_token )
	return access_token


def recognise_handwriting_pic(access_token,image_path):
	image = get_file_content(image_path)
	r = requests.post(
		url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token='+access_token,
		headers={"Content-Type":"application/x-www-form-urlencoded"},
		data = {'image':image})
	#print(r.text)
	j = json.loads(r.text)
	words_result = j.get('words_result')
	for i in words_result:
		print(i.get('words'))
		word = i.get('words')
	return word

access_token = get_access_token()  # 获取一次保存下来就够了，一般1个月有效期

def getResult():
	return recognise_handwriting_pic(access_token,image_path='E:\pycharm\ex2\static\img\icon.jpg')
