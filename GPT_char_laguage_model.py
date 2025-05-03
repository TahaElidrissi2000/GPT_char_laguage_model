# Libraires 
import requests
from pathlib import Path
import torch
import torch.nn as nn 
import sys


# Hyperparametrs 
batch_size=32 # how many indepent sequences will we process in paralelle ?
block_size=8 # the maximum lenght of the contexte we are going to use
epochs = 50
eval_epochs=10
n_emb = 32 # add this dimension to  make the token embedding as an intermediate phase  
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Print all values
print(f"Hyperparameters:")
print(f"  Batch size     : {batch_size}")
print(f"  Block size     : {block_size}")
print(f"  Epochs         : {epochs}")
print(f"  Eval every     : {eval_epochs} epochs")
print(f"  Embedding dim  : {n_emb}")
print(f"  Device         : {device} \n")

"""
===================Line of code to doanlod the data set directly forl github repo===============================
# Setup path to a data folder
data_path=Path("Pyscript/data/")

# If the image folder doesn't exist, download it and prepare it ...
if data_path.is_dir():
  print(f"{data_path} directory already exists .... skipping download")

else:
  print(f"{data_path} does not exist .... creating one ")
  data_path.mkdir(parents=True,exist_ok=True)
  # Dowload the data
  with open(data_path / "Tiny Shakespeare","wb") as f:
    request=requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    print("Downloading the data ....\n")
    f.write(request.content)
===================Line of code to doanlod the data set directly forl github repo===============================
    
"""




# Upload the data on the text variable 
with open("Pyscript/data/Tiny Shakespeare","r",encoding="utf-8") as f:
  print("Reading the data ....\n")
  text=f.read()


# Extract all the chars of our text and he vocab size  
chars = sorted(list(set(text)))


vocab_size = len(chars)

# Fucntion to go from int to chars and vice-versa : 
stoi={ch:i for i,ch in enumerate(chars)}
itos ={i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s ]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode our data into int to use later during training :
data =torch.tensor(encode(text),dtype=torch.long)

# Split the data into training and testing data :
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# Return a single batch from the selected data dataset 
def get_batch(split):
  data = train_data if split=="train" else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in ix ])
  y=torch.stack([data[i+1:i+block_size+1] for i in ix ])
  return x,y


# Bulding the model that we strated with 
class BiagramLanguageModel(nn.Module):
  
  def __init__(self,vocab_size):
    super().__init__()
    # each token is associted to a representetive tensor according to his index 
    self.token_embeding_table =nn.Embedding(vocab_size,n_emb)
    # each token is associted to a representetive tensor according to his position 
    self.positional_embeding_table = nn.Embedding(block_size,n_emb)
    # A linear layer to process the embedding  
    self.lm_head = nn.Linear(n_emb,vocab_size)
    # A weighted matrix controlling the inluence of each char of the context on the next predicted token 
    self.attention_heads = Multi_Attention_Head(4,n_emb//4) # we want to keep the same number of channels after concatenation
    # each head will give us 8 channels that after concatenation become 32 channels  
    # Add the ffd layers :
    self.ffd = feedforward(n_emb)

  def forward(self,idx,targets=None):
    # idx represent Features (B,T)
    B,T = idx.shape
    # Retrieving the token/pos embeddings of each token 
    token_embedings = self.token_embeding_table(idx) # -> (B,T,C)
    positional_embeding = self.positional_embeding_table(torch.arange(0,T,device=device)) # (B,T)
    # Without explicitly creating a positional emb matrix , the broadcasting takes care of it 
    x = token_embedings + positional_embeding
    # Applying ghte attention block 
    x = self.attention_heads(x)
    # Applying the feed forward layer 
    x = self.ffd(x)

    logits = self.lm_head(x)
    
    # To calculate the loss we use cross entropy however the format required by pytorch:
    # is (B*T,C) where all the elements are streatched into 1 dimensional sequence
    if targets is None:
      loss=None
    else:
      # The targets are also of shape (B,T) and we want theme to be (B*T)
      B,T,C=logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T)
      loss=nn.functional.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,idx,max_new_tokens):
    # the job of thei fucntion is to generate the next tokens following the contexte in (B,T)
    # Idx is a (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      #crop idx to the last block_size tokens 
      crop_idx = idx[:,-block_size:]
      #get the predictions
      logits,loss=self(crop_idx)
      #focus only on the last time step
      logits=logits[:,-1,:] #become (B,C)
      #apply softmax to get probailities
      probs=nn.functional.softmax(logits,dim=-1) #(B,C)
      #sample from the distribution
      idx_next=torch.multinomial(probs,num_samples=1)# (B,1) one prediction for what comes next
      #append sampled index to the running sequence
      idx=torch.cat((idx,idx_next),dim=1) #returns (B,T+1)
    return idx
  
def evaluation(model : nn.Module, splits : list[str], eval_epochs : int):
  # This function will evaluate the model to the training 
  out = {}
  with torch.inference_mode():
    model.eval()
    for split in splits :
      losses = torch.zeros((eval_epochs))
      for k in range(eval_epochs):
        x,y = get_batch(split)
        logits,loss = model(x,y)
        losses[k]=loss.item()
      out[split]=losses.mean()
    model.train()
    return out 

# ==================================SELF ATTENTION BLOCK=========================================       
# Creating the head module ,
class Attention_head(nn.Module) :
  def __init__(self,head_size) :
    super().__init__()
    self.value = nn.Linear(n_emb,head_size,bias=False)
    self.key = nn.Linear(n_emb,head_size,bias=False)
    self.query = nn.Linear(n_emb,head_size,bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    
  def forward(self,x) :
    B,T,C = x.shape 
    value = self.value(x) # (B,T,head_size)
    key = self.key(x) # (B,T,head_size)
    query = self.query(x) # (B,T,head_size)
    wei = key @ query.permute(0,2,1) * (C **-0.5) # (B,T,head_size) @ (B,head_size,T) - > (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei = torch.softmax(wei,dim=1)
    return wei @ value

class Multi_Attention_Head(nn.Module):
    def __init__(self,nb_heads,hd_size) :
      super().__init__()
      self.List_Attention_Heads = nn.ModuleList(Attention_head(hd_size) for i in range(nb_heads))
    
    def forward(self,x):
      return torch.cat([h(x) for h in self.List_Attention_Heads],dim=-1)

class feedforward(nn.Module):
  # This layer allow the attention block results to be processed before genreting the logits 
  def __init__(self,in_channels):
    super().__init__()
    self.out = nn.Sequential(
      nn.Linear(in_channels,in_channels),
      nn.ReLU()
    )

  def forward(self,x):
    return self.out(x)
# ==================================SELF ATTENTION BLOCK========================================= 



    
     

# ==================================TRAINING LOOP========================================= 
model = BiagramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)


for epoch in range(epochs):
  # Forward
  #Select different values each epoch
  xb,yb=get_batch("train")
  out,loss = model(xb,yb)

  
  # print out the perf 
  if epoch % eval_epochs ==0:
    out_perfomance = evaluation(model,["train","val_data"],eval_epochs)
    print(f"Step : {epoch} | Train loss : {out_perfomance['train']} | Val loss : {out_perfomance['val_data']}")
  # Backawrd prob
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

out_perfomance = evaluation(model,["train","val_data"],eval_epochs)
print(f"Step : {epoch} | Train loss : {out_perfomance['train']} | Val loss : {out_perfomance['val_data']}")
# ==================================TRAINING LOOP========================================= 




# ================Evaluating the model ====================================
context = torch.zeros((1,1),dtype=torch.long,device = device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))
# ================Evaluating the model ====================================