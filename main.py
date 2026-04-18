import os
import io
import asyncio
import uvicorn
import torch
import clip
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional

# 全局状态字典，用于存储常驻内存的模型和设备标识
ml_models = {}

# 全局异步锁，⚠️ 用于限制并发，防止显存溢出或 CPU 资源被瞬间耗尽
gpu_lock = asyncio.Lock()

# ⚠️ 严谨验证：读取环境变量，默认加载 512 维的轻量级模型
MODEL_DIM_ENV = os.environ.get("CLIP_MODEL_TYPE", "512")
if MODEL_DIM_ENV == "768":
    MODEL_NAME = "ViT-L/14"
    MAX_NATIVE_DIM = 768
# ⚠️ 致命类型错误已精准修复：将整数 512 改为字符串 "512"，防止环境变量匹配失效坠入 RN50 分支
elif MODEL_DIM_ENV == "512":
    MODEL_NAME = "ViT-B/32"
    MAX_NATIVE_DIM = 512
else:
    MODEL_NAME = "RN50"
    MAX_NATIVE_DIM = 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务生命周期管理器：在服务启动时执行资源加载，在服务关闭时清理。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"初始化服务... 正在根据环境变量加载单例 CLIP 模型 ({MODEL_NAME}, 原生维度: {MAX_NATIVE_DIM}) 到 {device} 计算节点")

    # 单例加载模型并存入全局字典，极大节省内存
    model, preprocess = clip.load(MODEL_NAME, device=device)
    ml_models["model"] = model
    ml_models["preprocess"] = preprocess
    ml_models["device"] = device
    ml_models["max_dim"] = MAX_NATIVE_DIM

    print("✅ 模型加载完成，FastAPI 服务已就绪！")
    yield  # 交出控制权，服务开始持续阻塞运行

    # 停止时的清理逻辑
    print("正在关闭服务，释放资源...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 实例化 FastAPI 对象
app = FastAPI(
    title=f"CLIP Image & Text Embedding API ({MAX_NATIVE_DIM}D)",
    description="提供基于 CLIP 模型的图文向量提取服务，支持外部参数降维及图文零样本匹配。",
    version="1.4.0",
    lifespan=lifespan
)


# 定义接口的 JSON 返回格式规范
class EmbeddingResponse(BaseModel):
    status: str
    dimension: int
    embedding: List[float]


class TextEmbeddingRequest(BaseModel):
    text: str
    target_dim: Optional[int] = None  # 可选的外部降维参数


# 定义图文匹配结果的 JSON 规范
class MatchResult(BaseModel):
    text: str
    probability: float


class MatchResponse(BaseModel):
    status: str
    results: List[MatchResult]


@app.post("/api/v1/embedding/image", response_model=EmbeddingResponse)
async def generate_image_embedding(
        file: UploadFile = File(...),
        target_dim: Optional[int] = Query(None, description="⚠️ 可选参数：目标降维维度 (非 MRL 模型截断会降低精度)")
):
    """
    接收客户端上传的图像文件，并返回对应的特征向量
    """
    # 1. 基础校验：拦截非图像类型文件
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型非法，请上传 image/* 格式文件。")

    max_dim = ml_models["max_dim"]
    if target_dim is not None and target_dim > max_dim:
        raise HTTPException(status_code=400,
                            detail=f"请求维度 {target_dim} 超过了当前启动模型支持的最大原生维度 {max_dim}。")

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

                # 外部参数降维逻辑：执行切片截断
                if target_dim is not None and 0 < target_dim < image_features.shape[1]:
                    image_features = image_features[:, :target_dim]

                # L2 归一化 (为后续入库 Milvus 做余弦相似度检索准备，降维后必须重新归一化)
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


@app.post("/api/v1/embedding/text", response_model=EmbeddingResponse)
async def generate_text_embedding(request: TextEmbeddingRequest):
    """
    接收客户端传入的文本内容，并返回对应的特征向量
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空。")

    max_dim = ml_models["max_dim"]
    if request.target_dim is not None and request.target_dim > max_dim:
        raise HTTPException(status_code=400,
                            detail=f"请求维度 {request.target_dim} 超过了当前启动模型支持的最大原生维度 {max_dim}。")

    device = ml_models["device"]
    model = ml_models["model"]

    # 1. 文本预处理与 Tokenize
    try:
        # ⚠️ truncate=True 是绝对必须的，用于自动截断超过 77 Token 限制的超长文本，避免服务崩溃
        text_tensor = clip.tokenize([request.text], truncate=True).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本 Tokenize 处理失败: {str(e)}")

    # 2. 执行模型推理 (核心保护区)
    try:
        async with gpu_lock:
            with torch.no_grad():
                # 提取文本特征
                text_features = model.encode_text(text_tensor)

                # 外部参数降维逻辑：执行切片截断
                if request.target_dim is not None and 0 < request.target_dim < text_features.shape[1]:
                    text_features = text_features[:, :request.target_dim]

                # L2 归一化 (为后续入库 Milvus 做余弦相似度检索准备)
                text_features /= text_features.norm(dim=-1, keepdim=True)

        # 释放锁后，将 Tensor 转移到 CPU 内存，并转化为标准 Python 列表
        embedding_list = text_features.cpu().numpy().flatten().tolist()

        return EmbeddingResponse(
            status="success",
            dimension=len(embedding_list),
            embedding=embedding_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型文本前向传播推理失败: {str(e)}")


@app.post("/api/v1/match/image_text", response_model=MatchResponse)
async def match_image_and_text(
        file: UploadFile = File(...),
        texts: str = Form(..., description="逗号分隔的候选文本列表，例如：a dog, a cat, a car")
):
    """
    接收一张图片和一组候选文本，返回各个文本与图片的匹配概率（零样本分类）。
    内部调用 model(image, text) 前向交叉传播。
    """
    # 1. 基础校验
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型非法，请上传 image/* 格式文件。")
    if not texts or not texts.strip():
        raise HTTPException(status_code=400, detail="候选文本列表不能为空。")

    # 清洗并提取文本列表
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    if not text_list:
        raise HTTPException(status_code=400, detail="解析不到有效的候选文本。")

    # 2. 图像读取与格式规范化
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="文件内容破损或并非有效的图像文件。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器读取图像流失败: {str(e)}")

    device = ml_models["device"]
    model = ml_models["model"]
    preprocess = ml_models["preprocess"]

    # 3. 数据预处理
    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像张量预处理失败: {str(e)}")

    try:
        # ⚠️ 对传入的所有候选标签进行批量 tokenize，自动截断超长标签
        text_tokens = clip.tokenize(text_list, truncate=True).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本 Tokenize 处理失败: {str(e)}")

    # 4. 执行交叉前向传播推理 (核心保护区)
    try:
        async with gpu_lock:
            with torch.no_grad():
                # model(image, text) 计算相似度 logits 对数几率
                logits_per_image, logits_per_text = model(image_tensor, text_tokens)
                
                # 沿文本维度计算 softmax，得出所有候选文本的概率分布 (总和为 1.0)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0].tolist()

        # 将文本与算出的概率打包，并按概率从高到低排序
        results = []
        for text_label, prob in zip(text_list, probs):
            results.append(MatchResult(text=text_label, probability=prob))
        
        results.sort(key=lambda x: x.probability, reverse=True)

        return MatchResponse(status="success", results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图文前向传播匹配失败: {str(e)}")


# ⚠️ 关键启动逻辑：此处为阻塞 Web 服务的命脉，绝对不可缺失
if __name__ == "__main__":
    # 为了配合浮动卡片 UI 的潜在超时限制，确保不更改任何其他配置项
    uvicorn.run(app, host="0.0.0.0", port=8000)
