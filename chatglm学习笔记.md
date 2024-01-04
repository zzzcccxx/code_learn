# chatglm3学习笔记

```
# fly-iot
https://www.bilibili.com/video/BV1gu4y1a7Ug/?p=2&spm_id_from=pageDriver
```

1、使用docker并映射到相应磁盘空间。

2、使用web ui。

3、使用autodl来租4090显卡，使用命令行来调用端口的8000，如报错将https改成http

```python
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "chatglm3-6b",
     "messages": [{"role": "user", "content": "北京景点"}],
     "temperature": 0.7
   }'
```

```
# 铭轩精选
https://b23.tv/4UYmDyH  
```

1、手把手从零开始chatchat。

2、实现如何将远程端口跑的服务映射到本地，并在本地打开运行在远程的webui。

3、实现chatchat的本地知识库（但自测会出现瞎回答的现象）。

```
# 新建文件夹X
https://b23.tv/IZvCXw2
```

1、使用chatglm3的命令行来进行调用，先运行api.py，后编写py程序来在终端窗口里得到返回的输入输出。

2、但是是chatglm2的，版本老，视频有些过时。

```
# 崩坏的领航员
https://b23.tv/WJ7Kid9
```

1、修改工具，实现自定义工具，音视频的返回。

2、使用fastgpt来做知识库。

3、fastgpt不是开源，需要花费，且需要使用docker。

```
# Rocky77
https://www.bilibili.com/video/BV1E94y1V7Lz/?spm_id_from=333.999.0.0&vd_source=822d1ebf6d323eb45fc71d11318c77f3
```

1、教如何初始化lfs，并使用魔法方法clone权重文件。
