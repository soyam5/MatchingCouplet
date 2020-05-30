#-*-coding:utf-8-*-
from aip import AipImageClassify

""" 你的 APPID AK SK """
APP_ID = '20103218'
API_KEY = 'sGsoiPGUCC2nyCcrlGN1pneF'
SECRET_KEY = 'ttXvP6pCGcMRVHewFEf1j0HxMi60XV9g'

client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('E:\pycharm\ex2\static\img\img1.jpg')

""" 调用通用物体识别 """
client.advancedGeneral(image);

""" 如果有可选参数 """
options = {}
options["baike_num"] = 5

""" 带参数调用通用物体识别 """
result=client.advancedGeneral(image, options)
print(result)
"""读取对联字典"""
f=open(r"dic.txt","r")
str=f.read()
temp=str.split(" ")
dic=[]
for i in temp:
    if i=="":
        pass
    else:
        dic.append(i)
f.close()
"""对联高频词过滤"""
keyword=[]
for i in range(result['result_num']):
    str=result["result"][i]["keyword"]
    print(str)
    for j in dic:
        if j in str :
            keyword.append(j)
keyword = list(set(keyword))
print(keyword)

def getKeyword():
    return keyword


