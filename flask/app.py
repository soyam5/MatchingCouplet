#from rnn_poetry import *
from flask import Flask, request,make_response,render_template, redirect, url_for
from werkzeug.utils import secure_filename # 使用这个是为了确保filename是安全的
from os import path
import coupletAPI
import HWRAPI
import imgAPI
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/more',methods=['POST', 'GET'])
def more():

    if request.method=='POST':
        ##获取来自网页上传的图片
        icon = request.files["icon"]
        img = request.files["img"]
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path, 'static/uploads/')
        ##保存来自网页上传的图片
        if (icon):
            file_name1 = upload_path + "icon.jpg"
            icon.save(file_name1)
            ##通过调用api获取word（str）
            word = HWRAPI.getResult()
        if (img):
            file_name2 = upload_path + "img.jpg"
            img.save(file_name2)
            ##通过调用api获取couplets（列表对象，列表中的每一个对象代表一副对联）
            couplets = coupletAPI.getCouplets()
            ##通过couplet[i]选取一副对联（包括center（横幅），first（首联），second（尾联））
            center=couplets[0].center
            first=couplets[0].first
            second = couplets[0].second
            print(center+','+first+','+second)
        return redirect(url_for("more"))
    ##word为要返回的手写识别的字符串（str），couplets为图像识别要返回的对联对象/
    return render_template("more.html")


if __name__ == '__main__':
    app.run()
