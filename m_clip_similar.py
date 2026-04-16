import os
import glob
import requests
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# ==========================================
# 1. 核心系统配置
# ==========================================
# 我们刚刚部署成功的本地 CPU CLIP 特征提取微服务
EMBEDDING_API_URL = "http://127.0.0.1:8001/api/v1/embedding/image"

# Milvus 数据库连接配置 (标准的本地 Docker 部署默认端口)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# Milvus 数据表 (Collection) 配置
COLLECTION_NAME = "local_image_search"
DIMENSION = 512  # CLIP ViT-B/32 的标准输出维度

# ⚠️ 本地存放待入库图片的文件夹路径（请确保该目录存在且包含图片）
IMAGE_FOLDER = "./images"


# ==========================================
# 2. 向量提取客户端引擎
# ==========================================
def get_embedding_from_api(image_path: str) -> list:
    """
    向本地 FastAPI 微服务发送图片，获取 512 维的浮点向量
    """
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(EMBEDDING_API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data["embedding"]
            else:
                raise ValueError(f"接口返回状态异常: {data}")
        else:
            raise RuntimeError(f"HTTP 请求失败, 状态码: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"提取 {image_path} 向量失败: {e}")
        return None


# ==========================================
# 3. Milvus 数据库管理引擎
# ==========================================
def init_milvus_collection() -> Collection:
    """
    连接 Milvus 并初始化/重置数据表结构
    """
    print(f"正在连接 Milvus 数据库 ({MILVUS_HOST}:{MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # 如果集合已经存在，为了演示数据的纯净性，先将其删除（生产环境中请谨慎操作）
    if utility.has_collection(COLLECTION_NAME):
        print(f"检测到历史集合 '{COLLECTION_NAME}'，正在清理...")
        utility.drop_collection(COLLECTION_NAME)

    # 定义 Schema
    fields = [
        # 主键 ID：交由 Milvus 内部自动递增生成 (auto_id=True)，极大地降低业务层的维护成本
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 存储图片的绝对或相对路径
        FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=1000),
        # 存储 512 维特征向量
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]

    schema = CollectionSchema(fields, description="Local Image Search Demo")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"✅ 集合 '{COLLECTION_NAME}' 创建成功！")

    return collection


def build_index_and_load(collection: Collection):
    """
    为向量字段创建索引，并加载到内存准备检索
    """
    # ⚠️ 注意：如果您的图片总数极少（例如不足 128 张），创建 IVF_FLAT 索引中的 nlist=128
    # 可能会在后台触发聚类警告数据量不足，但这绝对不会影响程序运行和最终的精准搜索结果。
    index_params = {
        "metric_type": "L2",  # 采用欧氏距离测量相似度
        "index_type": "IVF_FLAT",  # 倒排文件索引，内存和速度的良好平衡
        "params": {"nlist": 128}
    }
    print("正在为向量字段创建 IVF_FLAT 索引...")
    collection.create_index(field_name="embedding", index_params=index_params)

    print("正在将集合加载至高速内存...")
    collection.load()
    print("✅ 索引构建与内存加载完成！")


# ==========================================
# 4. 业务流水线逻辑
# ==========================================
def process_and_insert_images(collection: Collection):
    """
    遍历本地文件夹，调用接口提取向量并存入 Milvus
    """
    # 匹配文件夹下所有的 jpg, jpeg, png 图片
    search_patterns = [os.path.join(IMAGE_FOLDER, "*.jpg"),
                       os.path.join(IMAGE_FOLDER, "*.jpeg"),
                       os.path.join(IMAGE_FOLDER, "*.png")]

    image_paths = []
    for pattern in search_patterns:
        image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print(f"⚠️ 警告：在 '{IMAGE_FOLDER}' 目录下未找到任何图片文件！")
        return

    print(f"扫描到 {len(image_paths)} 张图片，开始逐一提取特征入库...")

    valid_filepaths = []
    valid_embeddings = []

    for idx, path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] 处理中: {path}")
        vector = get_embedding_from_api(path)
        if vector is not None:
            valid_filepaths.append(path)
            valid_embeddings.append(vector)

    if valid_filepaths:
        # 由于我们设置了 auto_id=True，插入时只需传入 [非主键列1数据列表, 非主键列2数据列表]
        insert_data = [valid_filepaths, valid_embeddings]
        mr = collection.insert(insert_data)
        # 强制刷盘，确保数据立即可见
        collection.flush()
        print(f"✅ 成功将 {len(mr.primary_keys)} 条数据写入 Milvus！")
    else:
        print("❌ 所有图片特征提取均失败，无数据入库。")


def search_similar_images(collection: Collection, query_image_path: str, top_k: int = 3):
    """
    传入一张查询图片，在 Milvus 中检索最相似的 Top-K 张图片
    """
    print(f"\n==========================================")
    print(f"开始执行以图搜图检索，目标图片: {query_image_path}")

    # 1. 对查询图片自身提取特征向量
    query_vector = get_embedding_from_api(query_image_path)
    if query_vector is None:
        print("❌ 查询图片特征提取失败，中止搜索。")
        return

    # 2. 在 Milvus 中执行向量搜索
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}  # 决定搜索精度的参数，数值越大越精准，但稍微增加耗时
    }

    print(f"正在 Milvus 中比对，寻找最相似的 {top_k} 张图片...")
    results = collection.search(
        data=[query_vector],  # 待检索的向量列表
        anns_field="embedding",  # 被检索的字段
        param=search_params,  # 检索算法参数
        limit=top_k,  # 返回 Top K
        expr=None,  # 标量过滤表达式 (此处不使用)
        output_fields=["filepath"]  # ⚠️ 严谨要求：必须指定输出 filepath，否则默认只返回 ID 和距离
    )

    print("==========================================")
    print("检索结果排名：")
    for hits in results:
        for i, hit in enumerate(hits):
            # hit.distance 代表 L2 距离，越小说明越相似
            # hit.entity.get('filepath') 获取我们事先存入的自定义元数据
            print(f"Top {i + 1} | 距离: {hit.distance:.4f} | 匹配文件: {hit.entity.get('filepath')}")
    print("==========================================\n")


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 确保图片文件夹存在
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"⚠️ 已自动创建空目录 '{IMAGE_FOLDER}'，请放入一些测试图片后再运行本脚本。")
    else:
        # ⚠️ 精准修复：取消注释，恢复完整的入库与加载闭环逻辑

        # 2. 初始化 Milvus (此操作会彻底清空并重置历史表)
        collection = init_milvus_collection()

        # 3. 批量入库 (必须有数据存入才能搜索)
        process_and_insert_images(collection)

        # 4. 构建索引并加载内存 (⚠️ 核心关键步骤：必须调用此函数内的 collection.load() 才能避免报错)
        build_index_and_load(collection)

        # 5. 测试检索 (动态选择已入库的最后一张图作为查询目标进行测试)
        test_images = glob.glob(r"/home/jhc/PyCharmMiscProject/clip/CLIP.png")
        if test_images:
            test_query_img = test_images[-1]
            search_similar_images(collection, query_image_path=test_query_img, top_k=3)