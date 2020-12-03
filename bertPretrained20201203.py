# -*- coding: utf-8 -*-


import torch
from transformers import BertModel, BertTokenizer, BertConfig
import os
import torch.nn as nn
import torch.optim as optim
from model20201126 import DeformerConfig,Deformer
import data20201124 as loader
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np

class bertSim(nn.Module):
    
    def __init__(self, bert_model_config):

        super(bertSim, self).__init__() 

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.config=bert_model_config
        
        self.config.output_hidden_states=True

        self.classifier = nn.Linear(bert_model_config.hidden_size, 2)

    def forward(self,input_ids,token_type_ids,attention_mask):
               
        output=self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,output_hidden_states=True)

        result=[]
     
        predict = self.classifier(output[1])

        result.append(predict)

        if self.config.output_hidden_states:
            result.append(output[2])
    
        #return predict
        return result

# 零填充
def padding(maxlen,text):
    for i in range (len(text)):
        if len(text[i]['input_ids'])<maxlen:
            tempinput=[0]*(maxlen-len(text[i]['input_ids']))
            temptoken=[1]*(maxlen-len(text[i]['input_ids']))
            text[i]['input_ids']=text[i]['input_ids']+tempinput
            text[i]['token_type_ids']=text[i]['token_type_ids']+temptoken
            text[i]['attention_mask']=text[i]['attention_mask']+tempinput
        elif len(text[i]['input_ids'])>maxlen:
            text[i]['input_ids']=text[i]['input_ids'][0:maxlen]
            text[i]['token_type_ids']=text[i]['token_type_ids'][0:maxlen]
            text[i]['attention_mask']=text[i]['attention_mask'][0:maxlen]
    return text

#得出结果
def getPredict(predict):
    softmax=nn.Softmax(dim=1)
    predict=softmax(predict)
    predict_result=[]
    for i in range(len(predict)):
        if predict[i][0]>predict[i][1]:
            predict_result.append(0)
        else:
            predict_result.append(1)  
    return predict_result

# 计算准确度
def accuracy(predict,target):
    predict=getPredict(predict)
    predict=torch.LongTensor(predict)
    count=0
    for i in range(len(target)):
        if predict[i]==target[i]:
            count+=1
    return count/len(target)
    
data_name='quora'
config=DeformerConfig()

#load data
train,test,dev=loader.load_data(data_name)
print('finish loading data')

#encode data
#train_Text,train_target=loader.encode_data_plus(train)
test_Text,test_target=loader.encode_data_plus(test)
#dev_Text,dev_target=loader.encode_data_plus(dev)
print('finish encoding data')

#zero padding
#train_Text=loader.padding_plus(config.maxlen_plus,train_Text)
test_Text=loader.padding_plus(config.maxlen_plus,test_Text)
#dev_Text=loader.padding_plus(config.maxlen_plus,dev_Text)
print('finish zero padding')

#get input_ids,token_type_ids,attention_mask
#train_input_ids,train_token_type_ids,train_attention_mask=loader.get_ids(train_Text)
test_input_ids,test_token_type_ids,test_attention_mask=loader.get_ids(test_Text)
#dev_input_ids,dev_token_type_ids,dev_attention_mask=loader.get_ids(dev_Text)
print('finish decomposing')

#to tensor
#train_input_ids = torch.LongTensor(train_input_ids)
#train_token_type_ids = torch.LongTensor(train_token_type_ids)
#train_attention_mask = torch.LongTensor(train_attention_mask)
#train_target=torch.LongTensor(train_target)

test_input_ids = torch.LongTensor(test_input_ids)
test_token_type_ids = torch.LongTensor(test_token_type_ids)
test_attention_mask = torch.LongTensor(test_attention_mask)
test_target=torch.LongTensor(test_target)

#dev_input_ids = torch.LongTensor(dev_input_ids)
#dev_token_type_ids = torch.LongTensor(dev_token_type_ids)
#dev_attention_mask = torch.LongTensor(dev_attention_mask)
#dev_target=torch.LongTensor(dev_target)
print('finish to tensor')

#to Dataset
#train_dataset=TensorDataset(train_input_ids,train_token_type_ids,train_attention_mask,train_target)
test_dataset=TensorDataset(test_input_ids,test_token_type_ids,test_attention_mask,test_target)
#dev_dataset=TensorDataset(dev_input_ids,dev_token_type_ids,dev_attention_mask,dev_target)
print('finish to dataset')

#to DataLoader
#train_loader=DataLoader(dataset=train_dataset,
#                        batch_size=100,
#                        shuffle=False,
#                        num_workers=2)
test_loader=DataLoader(dataset=test_dataset,
                        batch_size=50,
                        shuffle=False,
                        num_workers=2)
#dev_loader=DataLoader(dataset=dev_dataset,
#                        batch_size=100,
#                        shuffle=False,
#                        num_workers=2)
print('finish to dataloader')

#定义模型
bertConfig=BertConfig()
bert = bertSim(bertConfig)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Let's use", torch.cuda.device_count(), "GPUs")
print(device)
bert = nn.DataParallel(bert)
bert.to(device)
criterion = nn.CrossEntropyLoss()#交叉熵损失
optimizer = optim.Adam(bert.parameters(), lr=0.0001)
kl_outputs=[]
hidden_states=[]

#训练模型
for epoch in range(1):
    #for i, data in enumerate(train_loader):
    for i, data in enumerate(test_loader):
        
      train_input_ids,train_token_type_ids,train_attention_mask,train_target=data
      train_input_ids,train_token_type_ids,train_attention_mask,train_target=Variable(train_input_ids),Variable(train_token_type_ids),Variable(train_attention_mask),Variable(train_target)
      train_input_ids=train_input_ids.data.cuda()
      train_token_type_ids=train_token_type_ids.data.cuda()
      train_target=train_target.data.cuda()
      outputs=bert(train_input_ids,train_token_type_ids,train_attention_mask)
      
      output=outputs[0]
      hidden_state_output=outputs[1]
    
      hidden_state=[]
      if epoch==0:
          kl_outputs.append(output.to('cpu').detach().numpy())
          for j in range(config.upper_layer_num):
              hidden_state.append(hidden_state_output[j+config.lower_layer_num+1].to('cpu').detach().numpy())
          hidden_states.append(hidden_state)
      loss = criterion(output, train_target) # for sentence classification
      
      train_target=train_target.to('cpu')
      acc=accuracy(output,train_target)
      
      print('Epoch:',epoch,'| Step:',i,'| loss:','{:.6f}'.format(loss),
              '| accuracy:',acc) 
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
print("-----------------------------")
print("......finish train model.....")
print("-----------------------------")
kl_outputs=np.array(kl_outputs)
np.savez('/kl_outputs.npz',kl_outputs)
print(kl_outputs.shape)
print('finish save kl_outputs')
hidden_states=np.array(hidden_states)
np.savez('hidden_states.npz',hidden_states)
print(hidden_states.shape)
print('finish save hidden_states')