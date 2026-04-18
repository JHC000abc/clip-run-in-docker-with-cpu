# 1. 采用最小化的 Python 3.9 Debian Slim 镜像
# 底层仅包含百兆级别的基础系统库，彻底摒弃臃肿的 CUDA 环境
FROM python:3.9-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 安装系统依赖：Git（用于从 GitHub 源码构建 CLIP）
# ⚠️ slim 镜像非常精简，默认不带 git。安装后立刻清理 apt 缓存以追求极致缩减体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 4. 优先单独安装纯 CPU 版本的 PyTorch (核心瘦身步骤)
# ⚠️ 必须通过 --index-url 强制指定 cpu 专用下载通道，避免拉取数 GB 的默认 GPU 版本
RUN pip install --no-cache-dir uv --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --system

# 5. 复制依赖清单并安装其余依赖
# ⚠️ 附加了清华大学 pip 国内镜像源，防止在无科学上网的环境下安装常规包时发生 i/o timeout
COPY requirements.txt .
RUN uv pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --system

# 7. 复制本地的业务代码到容器中
COPY main.py .

# 8. 声明服务端口
EXPOSE 8000

# 9. 定义容器启动时的默认执行命令
CMD ["python", "main.py"]
