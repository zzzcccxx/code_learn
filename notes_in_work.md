# 在工作中的笔记

### 1. 出现sh xxx.sh不能正常运行或显示某个已安装的包未安装。

检查source环境。

### 2. 如何将数组中的所有元素整合成一个字符串？

` middle = " ".join(list(a)) `

### 3. 去掉字符串首尾的空格？

`a.strip()`

### 4. 只有字符串才能.split(“xx”)

### 5. 若读取某文件夹下所有文件名，用os.wlak(dic)

`for i,j,k in os.walk(filePath):`

### 6. 在列表中所有字符串前都加某个字符  c

`b = ['c' + str(i) for i in a]`

### 7. pandas中只显示某几行或某几列

`pd.set_option选项调整`

### 8. 如何只显示字典的前5个key_value

`print(list(字典名.items())[:5])`

### 9. can’t convert CUDA:0 device type tensor to numpy, use Tensor.cpu()….

在使用np.argmax时传入的变量必须是cpu上的而不能是gpu的

### 10. 使用qid的embedding层大小(a,b)

其中a为卷子名称使用sklearn进行LabelEncoder再去重的数量；
b为隐藏层神经元数量。

### 11. 使用pandas无法替换字符(将a换成b)

```python
df['comment'] = df['comment'].str.replace("a","b")
df.to_csv('./~~~', index=False)
```

### 12. pandas对某列实施某操作后形成新的列

```python
# 对paper列按字典查找其对应的答案，放入对应的答案列
df['answer'] = df['paper'].apply(lambda x: dic_answ[x])
```

```python
# 目标列为a丨丨丨b格式，单独生成一列b格式 
df = pd.read_csv('~~~')
df['new'] = df['目标列'].apply(lambda x: x.split('丨丨丨')[1])
df['new'] = df['new'].str.replace("<void>","")
df.to_csv('~~~', index=False)
```

### 13. 将列表进行拼接并生成一个大字符串

```python
text_conb = ['[cls]'] + [text1] + ['[sep]'] + [text2] + ['[sep]']
text_conb = ' '.join(map(str,text_conb))
```

### 14. CUDA error:xxxxxxx

使用cpu进行调试，能找到真正问题

### 15.  txt文件按\t来进行分列

`df = pd.read_csv('~~', sep='\t')`

### 16. 提取出pandas包含某个字符的行

`print(df[df['name'].str.contains('@@@@')])`

### 17. K折交叉验证

```python
import pandas,sklearn,os
train_file = '~~~'
cv_file = '~~~'
train_df = pd.read_csv('~~')
aa = train_df[1:]
folds = KFold(n_splits=5, shuffle=True,random_state=66)
for fold_,(train_idx,val_idx) in enumerate(folds.split(aa)):
    train_data=aa.iloc[train_idx]
    val_data=aa.iloc[val_idx]
    train_path=os.path.join('train_cross_valid',fold_)
    val_path=os.path.join('val_cross_valid',fold_)
    train_data.to_csv(train_path,index=False)
    val_data.to_csv(val_path,index=False)
```

### 18. 如何遍历字典中的每个值

```python
for key in dicta.keys():
    dicta[key] = list(filter(lambda x: x!='丨'and x!="", dica[key].split('_')))
```

### 19. 读取json文件时出现str object has no attribute read

用load()或loads()或

```python
import json
_lines = json.load(open(path))
```

### 20. 生成和合并json文件

```python
# 生成generate.py
import json
import pandas as pd
dic =
[
    { 
     "id": "@@@@@@@@@",
     "answer":"answer1",
     "text":"text1",
     "human_score":0.25
    },
    { 
     "id": "#########",
     "answer":"answer2",
     "text":"text2",
     "human_score":0.35
    },
    { 
     "id": "$$$$$$$$",
     "answer":"answer3",
     "text":"text3",
     "human_score":0.45
    }       
]
with open("/root/learn_code/data/json3.json", "w") as fp:
    json.dump(dic, fp, indent=2)
fp.close()
```

```python
# 合并法1
import json

json_list = ["/root/learn_code/data/json2.json", "/root/learn_code/data/json3.json"]

with open('/root/learn_code/data/json1.json') as f1:
  data1 = json.load(f1)

for j_file in json_list:
    with open(j_file) as f:
        data = json.load(f)
        data1 += data

with open('/root/learn_code/data/chatgpt_result.json', 'w') as outfile:
    json.dump(data1, outfile, indent=2)



 # 合并法2
json_list = glob.glob(f"{root_path}/*.json")
data_list = []
for json_file in json_list:
    data_file = json.load(open(json_file))
    data_list.extend(data_file)
random.shuffle(data_list)
with open('xxx.json','w',encoding='utf-8') as pw:
    json.dump(data_list, pw, ensure_ascii=False, indent=2)
pw.close()
```

### 21. 把json文件格式转为可训练的dataset格式

```python
json_path = 'xxx.json'
_lines = json.load(open(json_path))
iters = itertools.chain(*[_lines])    # 将不同行解析为一行 
# iters[0]为{'id': '0000011', 'answer': 'answer1', 'text': 'text1', 'human_score': 0.25}
dataset = []
for line in iters:
    dataset.append(line)
```

### 22. 自定义huggingface模型

```python
config = AutoConfig.from_pretrained('xxxx')
encoder = AutoModel.from_pretrained('xxxx')
config.update({
    "output_hidden_states" = True,
    "num_labels" = 1,
    "add_pooling_layer" = False
})
```

### 23. 修改pandas将50记为49.999999错误

```python
gold=round(gold) if abs(gold-round(gold)) < 1e-4 else gold
```

### 24. 在list.lst中包含所有的.wav文件路径，pf_human_score.csv中包含所有考试的ID和人工分，将wav的ID在pf_human_score.csv中存在的路径复制到指定目录并创建human_score.csv

```python
wav = pd.read_csv('', header=None, names=['wav_id'])
wav['final_id'] = wav['wav_id'].apply(lambda x: ~~~)
os.makedirs('~', exist_ok=True)
H1 = open('~~', 'w')
H1.write('ID' + '\t' + 'human_score' + '\n')
score_file = pd.read_csv('~~', header=None, sep='\t', names=['aa', 'bb', 'cc'])
all_wav_id = set(score_file['ID'])
for i in range(len(wav['final_id'])-1):
    if wav['final_id'][i] in all_wav_id and ...:
        shutil.copy(~~, '~~' + '.wav')
        H1.write()
```

### 25. 若dataframe取出的某列后有索引，如：

```python
score = score_file[score_file['ID']==wav['final_id'][i]]['human_score']
# 35778        0.5
score = score.values.item()
# 0.5
```

### 26.解压.zip文件到指定目录

```python
unzip xx.zip -d 指定目录
```

### 27. 去重得到sklearn的LabelEncoder

```python
que_le = sklearn.preprocessing.LabelEncoder()
que_le.fit(df['paper_id'])
df['paper_id'] = list(que_le.transform(df['paper_id']))
```

### 28. 在shell指令中遍历

```
for dataset in bj tj sh hz 
do
case $dataset in
    bj)
    exam_type=1
    fenzhi=2
    ;;
    tj)
    exam_type=2
    fenzhi=5
    ;;
    sh)
    exam_type=2
    fenzhi=5
    ;;
    hz)
    exam_type=2
    fenzhi=5
    ;;
esac
path=$dataset/$exam_type/$fenzhi
done
```

### 29. 在云端运行程序得到网页的显示界面，如何打开http://127.0.0.1:5000/

```
curl http://127.0.0.1:5000/    #在终端打开网页
```

复制内容，在本地新建文本文件，粘贴，再将后缀改为html打开即可。

### 30. 使用同组其他人的环境

```python
source /dir/.bashrc_fairseq_cuda10
```

### 31. 如何从字典dic中随机取出字典的键来？

```python
random.choice(list(dic))
```

### 32. 如何查看huggingface中的特殊标志位？

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")
original = tokenizer(sentence1,sentence2, max_length=512, truncation=True, return_tensors='pt')
original = {k:v[0] for k,v in original.items()}
print(original)
print(tokenizer.special_tokens_map)
print(tokenizer.encode(['[CLS]', '[SEP]', 'PAD']))
print(tokenizer.decode([0, 2]))
```

### 33. 如何在输入近tokenizer的两端文本中前面的那段采用bert的mask机制来进行掩码？

```python
# method 2 
test1 = tokenizer4(sentence1, max_length=512,add_special_tokens=False, truncation=True, return_tensors='pt')
test1 = {k:v[0] for k,v in test1.items()}
test1 = test1['input_ids']
labels = test1.clone()
select_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()        # 选中15%
mask_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & select_indices        # 15%中的80%
test1[mask_replaced] = tokenizer4.convert_tokens_to_ids(tokenizer4.mask_token)
indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & select_indices & ~mask_replaced
random_words = torch.randint(len(tokenizer4), labels.shape, dtype=torch.long)    # 替换成词典中的随机词
test1[indices_random] = random_words[indices_random]
print(test1)
sentence_changed_id = test1
sentence_changed = tokenizer4.decode(sentence_changed_id.tolist())
print(sentence_changed)

model_input = tokenizer4(sentence_changed,sentence2, max_length=512, truncation=True, return_tensors='pt')
```

### 34.若想动态的看nvidia的显存变化，只能用在linux系统重

```
watch -n 3 nvidia-smi
```

### 35. 如何本地使用docker新建环境并映射到 /data磁盘

```
docker run -itd --name chatglm3 -v `pwd`/ChatGLM3:/data \
--gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
-p 8501:8501 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
```

### 36. docker启动和运行

```
docker start chatglm3

docker exec -it chatglm3 bash
```

### 37. 查看磁盘空间

```
du -sh chatglm3/
du -h --max-depth=1   # 只查看文件夹，不查看当前目录下文件夹中的每个文件。
```

### 38. 配置代理和清楚代理（梯子）

```
export http_proxy=http://172.31.233.190
export https_proxy=http://172.31.233.190


unset http_proxy
unset https_proxy
```

### 39. 服务器跑的生成端口怎么本地访问

```python
# 如服务器给出url为 http://0.0.0.0:8501
ssh -L 8501:localhost:8501 -p 30056 root@172.31.233.190
# 将服务器的8501与本地的8501做映射，代码前者为服务器的
```

### 40. python导入不了自己写的库怎么办

```
import sys
sys.path.append('包的路径')
```

### 41. 指定某个gpu运行程序

```
CUDA_VISIBLE_DEVICES=0 python xxx.py
```

### 42. 修改ssh远程密码

```
passwd    用户名
```

### 43. 使用autodl怎么使用端口：

参考帮助文档，先开本地，再将端口映射为6006.

### 44. 怎么从huggingface上初始化lfs

<<<<<<< HEAD

```
sudo apt-get install git-lfs
=======
```shell
sudo apt-get install git-lfs
或者
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
$ sudo yum install git-lfs
$ git lfs install

>>>>>>> 6a6dbf8298c1342d54fb3de5141f894b515f4c01
```

### 45. git无法使用conda命令

```
首先将anaconda路径和anaconda下的scripts路径添加到环境变量
再在Anaconda的安装位置处，例如D:\Anaconda\etc\profile.d，在profile.d文件夹中“右击”选择“Open Git Bash Here”。
输入echo ". '${PWD}'/conda.sh" >> ~/.bashrc，之后回车。
```

<<<<<<< HEAD

### 46. gitbash中环境激活不了且报一堆错误

=======

### 46. 如何动态查看日志

```shell
tail [-f -num] 文件路径
-f表示动态，num表示查看几行
```

# 本地git无法激活虚拟环境

```
export PYTHONUTF8=1
然后再激活
```

### 47. 虚拟机使用共享文件夹

```
https://zhuanlan.zhihu.com/p/650638983
添加之后记得使用
$ sudo mount -t fuse.vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
回在/mnt/hgfs中新建个/share文件夹用了存放共享文件
```

### 48. github如何查看自己参与过的issue

```
is:issue involves:zzzcccxx
```

### 49. 如何使用某个训好的小模型的前几层作为embedding模型

```python
### 先从modelscope或huggingface上下载模型，model.bin

### 先看模型都有哪几层
import torch

model = torch.load('pytorch_model.bin')

for name in model.keys():
    print(name)


### 将模型前面的embedding几层保存为新模型
embedding_layer = {}

# 提取嵌入层的参数
for key, value in model.items():
    if 'plm.embeddings' in key:
        embedding_layer[key] = value

torch.save(embedding_layer, './embedding_model.bin')


### 加载embedding模型并给出词嵌入表示
# 加载嵌入模型
embedding_model = torch.load('./embedding_model.bin')

with open('./vocab.txt', 'r') as f:
    vocab = [line.strip() for line in f]
    index_shoe = vocab.index('鞋')
    index_cabinet = vocab.index('柜')

# 创建一个包含"鞋"和"柜"索引的张量
input_data = torch.tensor([[index_shoe, index_cabinet]])
word_embeddings = embedding_model['plm.embeddings.word_embeddings.weight']
word_embedded = torch.nn.functional.embedding(input_data, word_embeddings)

print(word_embedded)  # 打印"鞋柜"的词嵌入表示
```

```python
import torch
from torch.nn import functional as F
import numpy as np

PRODUCT_LIST = ['鞋柜', '书架', '窗', '床', '门']
embedding_model = torch.load('./embedding_model.bin')

def preprocess(text):
    # 在这里添加你的预处理步骤
    pass

def get_embedding(text):
    # 使用你的模型将文本转化为词嵌入表示
    words = list(text)
    with open('./vocab.txt', 'r') as f:
        vocab = [line.strip() for line in f]
        indices = [vocab.index(word) for word in words]
    input_data = torch.tensor([indices])
    word_embeddings = embedding_model['plm.embeddings.word_embeddings.weight']
    word_embedded = torch.nn.functional.embedding(input_data, word_embeddings)
    return word_embedded


def get_most_similar(user_input, product_names):

    # user_input = preprocess(user_input)
    user_embedding = get_embedding(user_input)

    product_embeddings = [get_embedding(name) for name in product_names]
    import pdb; pdb.set_trace()
    product_embeddings = torch.stack([e for e in product_embeddings])
    similarities = F.cosine_similarity(user_embedding.unsqueeze(0), product_embeddings[0])
    most_similar_index = torch.argmax(similarities)

    return product_names[most_similar_index]

# 测试
user_input = "组合柜"
print(get_most_similar(user_input, PRODUCT_LIST))
```

### 50. 怎么查看运行的cpu程序的id
``` ps -ef | grep 应用名称
