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

image = get_file_content('E:\pycharm\ex2\static\img\img2.jpg')

""" 调用通用物体识别 """
client.advancedGeneral(image);

""" 如果有可选参数 """
options = {}
options["baike_num"] = 5

""" 带参数调用通用物体识别 """
result=client.advancedGeneral(image, options)
print(result)
for i in range(result['result_num']):
    print(result["result"][i]["keyword"])