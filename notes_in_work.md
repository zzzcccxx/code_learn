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
iters = itertools.chain(*[_lines])	# 将不同行解析为一行 
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



# Fairseq训练流程

```
fairseq-train\
--save-dir \	# 模型保存路径
--user-dir \	# fairseq中data\model\task文件夹的路径
--task \		#任务名（注册的）
--task中自定义的参数 \
--finetune-from-model \	# xxxxx.pt
```

## task.py

1. 拿到参数
2. set_up_task(cls,args,**kwargs)
3. load_dataset(self,split,epoch=0,**kwargs)

## dataset.py

1. 先\__init__初始化

2. \__getitem__(self,index)

	```
	item = self.dataset.iloc[index]
	text = item['xxx']
	model_inputs = self.tokenizer(text,max_length=512,truncation=True,return_tensor='pt')
	model_inputs = {k:v[0], for k,v in model_inputs.item()}
	ret = {
			"id":~~,
			"aaa":~~,
			"bbb":~~~
			}
	ret = {**ret, **model_inputs}
	return ret
	```

3. collater(self, samples):

	```
	scores = torch.tensor([s['score'] for s in samples])
	collater_data = {
					"xxx":~~,
					"aaaa":~~,
					"net_input":{"input_ids":...., "~~":~~, "xxx":xxx}
					"target":scores
						}
	return collater_data
	```

	

## model.py

1. build_model(cls, args, task):

	```
	config = AutoConfig.from_pretrained("")
	encoder = AutoModel.from_pretrained("")
	config.update({"xx":xx, "x":x})
	return cls(args, encoder, task, config)
	```

2. \__init__(self, args, encoder_task, config)->None:

	```
	定义自己的网络如：
	self.encoder = encoder
	self.ques_emb = torch.nn.Embedding(a,b)
	```

3. forward(self, input_idx, attention_mask, **kwargs):

	```
	...
	logit = torch.sigmoid(logit)
	ret = {"encoder_out" : logit}
	return ret
	```

	
