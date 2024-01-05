

#设置python环境镜像
FROM python:3.9

#将<资源目录>/data/code/Intership的代码添加到/usr/local/code文件夹，/usr/local/code不需要新建（docker执行时自建）
ADD ./facenet_pytorch_api  /usr/local/code


# 设置/usr/local/code文件夹是工作目录
WORKDIR /usr/local/code


# 更新apt库文件
RUN apt-get update

# 升级setuptools和pip
RUN pip install -U setuptools pip -i https://pypi.tuna.tsinghua.edu.cn/simple


# 安装相应的python库
RUN pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U openmim
RUN mim install mmcv==2.1.0



# 启动服务
CMD ["python", "app.py"]