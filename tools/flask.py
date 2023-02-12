# -*- coding: utf-8 -*-
from flask import (Flask, abort, redirect, request, jsonify, make_response)
# from flask_script import Manager # 命令行参数接收 shell, runserver


app = Flask(__name__)
#manager = Manager(app=app)

# get start


@app.route("/")
def get_start():
    return "Hello World!"


# 重定向
@app.route("/index")
def hello_world():
    return redirect("https://www.baidu.com")


# http方法: GET, POST, PUT, DELETE
@app.route("/test_inference", methods=["POST"])
def apply():
    # 获取请求参数
    req = request.get_json()

    # 业务代码
    data = req.get()

    # 构造响应对象
    rsp = jsonify(
        orderId="",
        memberId=""
    )
    return rsp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
#    manager.run() # python run.py runserver -d -r -h 127.0.0.1 -p 5000
