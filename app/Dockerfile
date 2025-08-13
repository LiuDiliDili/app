FROM python:3.9.19

LABEL author="SiNan"

# 设置工作目录
WORKDIR /app

# 先复制requirements文件并安装依赖,创建必要的目录结构
COPY requirements.txt .
RUN pip install pip -U -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    && mkdir -p /app/model \
    && mkdir -p /app/logs

COPY . .

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python3", "main.py"]