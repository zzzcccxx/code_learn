# chatglm3学习笔记

```
https://www.bilibili.com/video/BV1gu4y1a7Ug/?p=2&spm_id_from=pageDriver
```

1、使用docker并映射到相应磁盘空间。

2、使用web ui。

3、使用autodl来租4090显卡，使用命令行来调用端口的8000，如报错将https改成http

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "chatglm3-6b",
     "messages": [{"role": "user", "content": "北京景点"}],
     "temperature": 0.7
   }'

```







### langchain-chatchat

