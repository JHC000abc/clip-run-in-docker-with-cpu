import io
import asyncio
import uvicorn
import torch
import clip
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

# 全局状态字典，用于存储常驻内存的模型和设备标识
ml_models = {}

# 全局异步锁，⚠️ 用于限制并发，防止显存溢出或 CPU 资源被瞬间耗尽
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务生命周期管理器：在服务启动时执行资源加载，在服务关闭时清理。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"初始化服务... 正在加载 CLIP 模型 (ViT-B/32) 到 {device} 计算节点")

    # 加载模型并存入全局字典
    model, preprocess = clip.load("ViT-B/32", device=device)
    ml_models["model"] = model
    ml_models["preprocess"] = preprocess
    ml_models["device"] = device

    print("✅ 模型加载完成，FastAPI 服务已就绪！")
    yield  # 交出控制权，服务开始持续阻塞运行

    # 停止时的清理逻辑
    print("正在关闭服务，释放资源...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 实例化 FastAPI 对象
app = FastAPI(
    title="CLIP Image Embedding API",
    description="提供基于 CLIP 模型的图像向量提取服务，支持单节点排队推理。",
    version="1.0.0",
    lifespan=lifespan
)


# 定义接口的 JSON 返回格式规范
class EmbeddingResponse(BaseModel):
    status: str
    dimension: int
    embedding: List[float]


@app.post("/api/v1/embedding/image", response_model=EmbeddingResponse)
async def generate_image_embedding(file: UploadFile = File(...)):
    """
    接收客户端上传的图像文件，并返回对应的特征向量
    """
    # 1. 基础校验：拦截非图像类型文件
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型非法，请上传 image/* 格式文件。")

    # 2. 图像读取与格式规范化
    try:
        content = await file.read()
        # ⚠️ 必须调用 .convert("RGB") 丢弃可能存在的 Alpha 透明通道，否则会导致模型张量计算报错
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="文件内容破损或并非有效的图像文件。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器读取图像流失败: {str(e)}")

    device = ml_models["device"]
    model = ml_models["model"]
    preprocess = ml_models["preprocess"]

    # 3. 图像预处理 (此步骤占用极低，不加锁)
    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像张量预处理失败: {str(e)}")

    # 4. 执行模型推理 (核心保护区)
    try:
        # 申请排他锁：确保当前时刻只有这一个张量在消耗核心资源
        async with gpu_lock:
            with torch.no_grad():
                # 提取图片特征
                image_features = model.encode_image(image_tensor)
                # L2 归一化 (为后续入库 Milvus 做余弦相似度检索准备)
                image_features /= image_features.norm(dim=-1, keepdim=True)

        # 释放锁后，将 Tensor 转移到 CPU 内存，并转化为标准 Python 列表
        embedding_list = image_features.cpu().numpy().flatten().tolist()

        return EmbeddingResponse(
            status="success",
            dimension=len(embedding_list),
            embedding=embedding_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型前向传播推理失败: {str(e)}")


# ⚠️ 关键启动逻辑：此处为阻塞 Web 服务的命脉，绝对不可缺失
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)