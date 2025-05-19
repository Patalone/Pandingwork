FROM python:3.10


# 设置容器中的工作目录
WORKDIR /app

# 拷贝核心文件和配置
COPY ./app.py /app/
COPY ./config.yml /app/
COPY ./api_config.yml /app/
COPY ./swagger_template.yml /app/
COPY ./requirements.txt /app/
COPY ./packages /app/packages
# 拷贝代码文件夹
COPY ./pan_intelligence_signal_core /app/pan_intelligence_signal_core
COPY ./configs /app/configs
COPY ./uploads /app/uploads

# 使用阿里云源
# 1. 升级pip
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip

# 2. 安装项目依赖
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --find-links=./packages -r requirements.txt

# 创建日志目录
RUN mkdir -p /app/logs

# 暴露端口
EXPOSE 5301

# 容器启动命令
CMD ["python", "app.py"]
