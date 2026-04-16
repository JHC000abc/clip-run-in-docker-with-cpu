# 为什么会有这个仓库那？
### 鄙人电脑比较落后，只有Nvidia 1050 显卡，试了很多方法，都没有办法运行CLIP，可能原因是硬件平台太老了，Nvidia不支持了，所以只能用cpu跑了，逛了一圈仓库，没发现太合适的，所以直接用Gemini手搓一个，自己用


```bash
docker build -t clip-api-cpu:v1 .
```

```bash
docker run -d --name clip-cpu-server -p 8001:8000 --restart always clip-api-cpu:v1
```
