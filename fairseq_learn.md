# Fairseq训练流程

```
fairseq-train\
--save-dir \    # 模型保存路径
--user-dir \    # fairseq中data\model\task文件夹的路径
--task \        #任务名（注册的）
--task中自定义的参数 \
--finetune-from-model \    # xxxxx.pt
```

## task.py

1. 拿到参数
2. set_up_task(cls,args,**kwargs)
3. load_dataset(self,split,epoch=0,**kwargs)

```python
def evaluate(self,result_triples,ckpt_name):
    pd.set_option("display.float_format", lambda x:f"{x:8.2f}")
    ckpt_name = ckpt_name.replace(".pt", "")
    result_path = os.path.join(self.args.result_path, f"result-{ckpt_name}.xlsx")
    items = []
    for _,sample,result in result_triples:
        pred = result['encoder_out'].tolist()*(最高分-最低分)+最低分
        gold = sample['target'].tolist()*(最高分-最低分)+最低分
        text = sample.get("text", '')
        item = [sample['key'], gold, pred, text]
        items.append(item)
    headers = ['~', '~', '~']
    ret = pd.DataFrame(items,columns=headers)
    ret.to_csv(...)
```

## dataset.py

1. 先__init__初始化

2. _*getitem_*(self,index)
   
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

2. _*init_*(self, args, encoder_task, config)->None:
   
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
