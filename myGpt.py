# %% [markdown]
# # 输入层
# sentence -> tokenizor -> padding -> embedding -> add position embedding
# -> masked attention

# %%
import tiktoken
import torch 
import torch.nn as nn
import numpy as np 
from torch.utils.data import  Dataset, DataLoader
import torch.optim as optim
encoding = tiktoken.get_encoding("gpt2")

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim = 256,head_num = 8,max_len = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.max_len = max_len
        self.q_layer = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.k_layer = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.v_layer = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.o_layer = nn.Linear(hidden_dim,hidden_dim)   # 标准 MHA 输出投影
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4,hidden_dim),
        )
        self.norm_layer1 = nn.LayerNorm(hidden_dim)
        self.norm_layer2 = nn.LayerNorm(hidden_dim)

        
    def forward(self,x):
        x,pad_mask = x
        h = self.norm_layer1(x)   # Pre-LN：norm 进子层，残差流保持干净
        q= self.q_layer(h).view(x.shape[0],x.shape[1],self.head_num,self.hidden_dim//self.head_num).transpose(1,2)
        k = self.k_layer(h).view(x.shape[0],x.shape[1],self.head_num,self.hidden_dim//self.head_num).transpose(1,2)
        v = self.v_layer(h).view(x.shape[0],x.shape[1],self.head_num,self.hidden_dim//self.head_num).transpose(1,2)
        attention_mask = torch.tril(torch.ones(x.shape[0],self.max_len,self.max_len,device=x.device))==0
        d_k = self.hidden_dim//self.head_num
        # 除以 sqrt(d_k)：不缩放的话 softmax 初始就饱和成 one-hot，梯度趋近 0
        attention_score = torch.softmax(torch.masked_fill(torch.masked_fill(q@k.transpose(2,3)/d_k**0.5,pad_mask.unsqueeze(1),-torch.inf),attention_mask.unsqueeze(1),-torch.inf),-1)
        attention_x = attention_score @v
        attention_x = self.o_layer(attention_x.transpose(1,2).reshape((x.shape[0],self.max_len,self.hidden_dim)))
        res_x = x + attention_x
        ff_2 = res_x + self.ff(self.norm_layer2(res_x))
        return (ff_2,pad_mask)
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim = 256,head_num = 8,layers = 8,vocab_size =10000,max_len = 16,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_embedding = torch.nn.Embedding(vocab_size,hidden_dim)
        self.pos_embedding = torch.nn.Embedding(max_len,hidden_dim)
        self.attention_layers = nn.Sequential(*[MultiHeadAttention(hidden_dim,head_num,max_len) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)   # 进 llm_head 前归一化残差流
    def forward(self,x,pad_mask):
        embedding = self.word_embedding(x) + self.pos_embedding(torch.arange(x.shape[1],device=x.device)).unsqueeze(0)
        out,_ = self.attention_layers((embedding,pad_mask))
        return self.final_norm(out)
class LLM(nn.Module):
     def __init__(self, hidden_dim = 256,head_num = 8,layers = 8,vocab_size =10000,max_len = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.decoder = TransformerDecoder(hidden_dim ,head_num ,layers ,vocab_size,max_len)
        # cross_entropy 内部自带 log_softmax，这里必须输出 logits，不能先做 Softmax
        self.llm_head = nn.Linear(hidden_dim,vocab_size)

     def forward(self,x,y,pad_mask):
         decoder_out = self.decoder(x,pad_mask)
         out = self.llm_head(decoder_out)
         loss = nn.functional.cross_entropy(out.view(-1,self.vocab_size),y.view(-1))
         return loss
     def generate(self,x,max_new_tokens = 16):
         valid_length = len(x)
         x = x + [0]*(self.max_len-len(x)) if self.max_len-len(x)>0 else x   # padding id=0，中英文词表均合法
         x= torch.tensor(x).cuda().view(1,-1)
         pad_mask = torch.ones(1,self.max_len,self.max_len)==0
         pad_mask = pad_mask.cuda()
         generate_text = []
         for i in range(max_new_tokens):
            pad_mask[:,:,valid_length:] = True
            decoder_out = self.decoder(x,pad_mask)
            out = self.llm_head(decoder_out)
            # next_token = torch.argmax(out,dim=-1)[0,valid_length-1]
            topk_vals, topk_idx = torch.topk(out[0,valid_length-1], 10)      # 只留前10个候选
            next_token = topk_idx[torch.multinomial(torch.softmax(topk_vals,dim=-1),1)]
    
            generate_text.append(next_token.item())
            if i+1!=max_new_tokens:
                x[0,valid_length] = next_token
                valid_length+=1

         return generate_text
         
            

# %%

class MyDataset(Dataset):
    def __init__(self, max_len = 16,use_cuda = False):
        self.pad_mask = torch.ones(max_len,max_len)==0
        self.x = []
        self.y = []
        with open("ch02\\01_main-chapter-code\\the-verdict.txt","r") as f:
            text = "".join(f.readlines())
            text_ids = encoding.encode(text)
            for i in range(0,len(text_ids)-max_len-1,5):
                self.x.append(text_ids[i:i+max_len])
                self.y.append(text_ids[i+1:i+max_len+1])
        self.use_cuda = use_cuda        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        if self.use_cuda:
            return torch.tensor(self.x[key]).cuda(), torch.tensor(self.y[key]).cuda(),    torch.tensor(self.pad_mask).cuda()
        else:
            return torch.tensor(self.x[key]), torch.tensor(self.y[key]),    torch.tensor(self.pad_mask)


# %%
class ChineseCharDataset(Dataset):
    """中文字符级数据集：每个字一个 id（红楼梦语料，词表约 4300）。
    不用 GPT-2 BPE：BPE 会把一个汉字切成多个 token，中文场景效率极低"""
    def __init__(self, max_len=128, use_cuda=False, stride=5, path="data\\hongloumeng.txt"):
        self.pad_mask = torch.ones(max_len, max_len) == 0
        text = open(path, encoding="utf-8").read()
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        ids = [self.stoi[c] for c in text]
        self.x = []
        self.y = []
        for i in range(0, len(ids) - max_len - 1, stride):
            self.x.append(ids[i:i + max_len])
            self.y.append(ids[i + 1:i + max_len + 1])
        self.use_cuda = use_cuda

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        if self.use_cuda:
            return torch.tensor(self.x[key]).cuda(), torch.tensor(self.y[key]).cuda(), self.pad_mask.cuda()
        return torch.tensor(self.x[key]), torch.tensor(self.y[key]), self.pad_mask

# %%

dataset = ChineseCharDataset(max_len=256, use_cuda=True)   # 换回英文语料: MyDataset(max_len=128, use_cuda=True)
dataLoader = DataLoader(dataset,32)
model = LLM(vocab_size=len(dataset.chars),max_len=256,layers=8).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
num_epochs = 100
total_steps = num_epochs * len(dataLoader)
warmup_steps = 500
import math
# warmup 防开局尖峰，cosine 后期衰减压低 loss
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: s/warmup_steps if s < warmup_steps
    else 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(total_steps-warmup_steps))))
step = 0
from collections import deque
loss_window = deque(maxlen=5)          # 单步波动大是正常现象，判断趋势看 100 步滑动平均
for i in range(0):
    for x,y,pad_mask in dataLoader:
        optimizer.zero_grad()
        loss  = model(x,y,pad_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_window.append(loss.item())
        step +=1
        if step % 50 == 0:
            print(f"Epoch：{i+1},step:{step},loss(100步均值):{sum(loss_window)/len(loss_window):.4f}")
    torch.save(model.state_dict(),f"checkPoints/cmodel.pth")

model.load_state_dict(torch.load(f"checkPoints/cmodel.pth"))
out = model.generate(dataset.encode("寶玉"),max_new_tokens=100)

print(dataset.decode(out))




# %%
