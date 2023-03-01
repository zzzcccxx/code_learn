# 口考历史数据训练大模型

## 数据预处理

### 原始数据

#### paper.txt

```
[choice]
1. I want to go shopping
2. I will go buy something
3. I'm going to the store
[keywords]
shopping丨store丨buy
[question]
what are you going to do?
[lmtex]
整次考试中所有卷子的正确答案
```

#### ars_result.csv

```
ppid	stu_id		text
xxxx	xxxx		xxxx
xxxx	xxxx		xxxx
xxxx	xxxx		xxxx
```

#### human_score.txt

```
ID		human_score
xxxx		xxx
xxxx		xxx
xxxx		xxx
```

---



### 数据处理

1. 通过map_to_phone和map_to_phone_index函数来生成词到音素和音素到idx的字典。
2. 得到paper_path_list:[‘卷1地址’, ‘卷2地址’,‘卷3地址’….]
3. 

```
调用get_all_answer_map函数传入(paper_path_list):
    调用get_extra_information函数来获得特殊字符转化信息
    调用get_qa_answer_map函数来获得整张卷子的answer和keywords并删除answer中的特殊字符:
        返回answer:{'卷1id':['答案1', '答案2'...], '卷2id':['答案1', '答案2'...], ...}
        返回keywords:{'卷1id':'关键句1丨关键句2丨关键句3', '卷2id':'关键句1丨关键句2丨关键句3'}
        返回question:{'卷1id':'问题'}
        返回replace_dice:{'12:30':'twelve thirty'...}
    将上述四个返回
answer_map: {'卷1id':[{'sentence':xxxx}]}
keywords = build_keywords_set(keywords) # 得到关键句中的所有单词去重并小写
keywords:{'卷1id':['关键词1','关键词2','关键词3'],'卷2id':['关键词1','关键词2','关键词3']...}
```

### 得到训练数据

```
进入get_alignment_result(middle_df, answer_map..., keywords)
遍历每张卷子:
    拿到某卷子对应的答案列表correct_answer_list:[{'sentence':A reporter}, {'sentence':The girl wants to be a reporter}]
    拿到某卷子的keyword['reporter', 'journalist']
    进入get_wer_and_dur函数(rec:识别的那句话, correct_answer_list, ..., keywords):
        拿到text_list['A reporter', 'The girl wants to be a reporter'], raw_align_script_list=[], wer_list=[]
        将rec整句话小写并按空格拆分赋值给rec
        遍历每个答案text_list:
            将对应的答案小写并按空格拆分得到ref, 并用nltk包分词性
            计算rec和ref的编辑距离得到metrix
            进入get_raw_align_scrip函数,传入(metrix, rec, ref, pos_bag):
                i是ref长度, j是rec长度
                score = [sys.maxsize for i in range(3)]
                edit_script = []
                while i>0 or j>0: 只要有没完的：
                    若由del得到,则score[0]=1
                    若由ins得到,则score[1]=1
                    若由sub得到,则score[2]=这个位置的rec和ref词的编辑距离matrix[-1][-1]
                    idx取score中最小的数的下标,若最小值都为max则edit_script.append(('EQUAL',query[j-1], reference[i-1],pos_bag[i-1][1]))
                    若idx为0则edit_script.append(('DEL','<void>', reference[i-1],pos_bag[i-1][1]))
                    若idx为1则edit_script.append(('INS',query[j-1], '<void>',pos_bag[i-1][1]))
                    若idx为2则edit_script.append(('SUB',query[j-1], reference[i-1], pos_bag[i-1][1]))
                    score = [sys.maxsize for i in range(3)]
                    edit_script.reverse()
                raw_align_script_list.append(edit_script)
                再经werdur函数返回wer:变换需要几步
                wer_list.append(wer)
        将wer_list从小到大排序，取变换所需步数最小的下标的描述raw_align_script_list[wer_list_index[0]]
        遍历最小步数的描述：comment_list=[]:
            若flag=item[0]为'EQUAL'则无操作
            若flag=item[0]为'SUB'：若(不)在关键词中则comment_list.append('(non-)keyword' + pos_bag(item[-1] + 'incorrect'))
            若flag=item[0]为'DEL'：若(不)在关键词中则comment_list.append('(non-)keyword' + pos_bag(item[-1] + 'missing'))
            comment_str = ','.join(comment_list)
        若len(comment_str)==0则comment_list.append('the answer is correct')
        data['comment'] = comment_list
        
```
