# Libraires 
import requests
from pathlib import Path
import torch
import torch.nn as nn 
import sys
import json 
import argparse


# Set up argument parsing
parser = argparse.ArgumentParser(description="Run model in different modes (train, evaluate, test).")
parser.add_argument("--config", type=str, default="launch.json", help="Path to config JSON file.")
parser.add_argument("--mode", type=str, choices=["train","debug"], default="train", help="Mode to run the model in.")
args = parser.parse_args()

# Load the config from the specified JSON file
with open(args.config, "r") as f:
    config = json.load(f)

# retriev the values corresponding to the selected lanching mde , either "debug" or "train"
mode_config = config.get(args.mode)


# Access config parameters
batch_size = mode_config.get("batch_size")  # Default to 32 if not provided
block_size = mode_config.get("block_size")  # Default to 8 if not provided
epochs = mode_config.get("epochs")  # Default to 50 if not provided
eval_epochs = mode_config.get("eval_epochs")  # Default to 10 if not provided
n_emb = mode_config.get("n_emb")  # Default to 32 if not provided

# Handle the device selection gracefully
device = mode_config["device"] if torch.cuda.is_available() else "cpu"  #  Default to GPU if available, else CPU

# Print out the configuration to confirm the loaded parameters
print(f"Configuration Loaded:")
print(f"  The selected mode is : {args.mode}")
print(f"  batch_size: {batch_size}")
print(f"  block_size: {block_size}")
print(f"  epochs: {epochs}")
print(f"  eval_epochs: {eval_epochs}")
print(f"  n_emb: {n_emb}")
print(f"  device: {device}")


"""
The lanching command is as follows :
    python GPT_char_language_model.py --config "launch.json" --mode "train" or "debug"

===================Line of code to download the dataset directly from a GitHub repository.===============================
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
===================Line of code to download the dataset directly from a GitHub repository.==============================
    
"""




# Upload the data on the text variable 
with open("Pyscript/data/Tiny Shakespeare","r",encoding="utf-8") as f:
  print("Reading the data ....\n")
  text=f.read()


# Extract all the chars of our text + define vocab size    
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map our chars to integers and vice-versa (for encoding and decoding purposes): 
stoi={ch:i for i,ch in enumerate(chars)}
itos ={i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s ]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode our text data into nuerical data:
data =torch.tensor(encode(text),dtype=torch.long)

# Split the data into training and testing data :
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# Return a single batch from the selected data dataset 

def get_batch(split):
    """
    Returns a single batch of data from the selected dataset (training or validation).

    Parameters:
    split (str): The dataset split to use. It can be either "train" or "val". 
    
    Returns:
    tuple: A tuple (x, y) where:
        - x (torch.Tensor): The input data batch with shape (batch_size, block_size).
        - y (torch.Tensor): The target data batch with shape (batch_size, block_size), 
                            which is the shifted version of x by one token.
    

    """
    
    # Select the correct dataset (train or validation)
    data = train_data if split == "train" else val_data

    # Randomly select starting indices for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Create input batch (x) by stacking sequences from the selected indices
    x = torch.stack([data[i:i + block_size] for i in ix])

    # Create target batch (y) by stacking shifted sequences from the selected indices
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


# Bulding the model that we strated with 
class BiagramLanguageModel(nn.Module):
  """
    
    A Biagram-based language model that uses token embeddings, positional embeddings, 
    multi-head attention, and feedforward layers to predict the next token in a sequence.

    Parameters:
    vocab_size (int): The size of the vocabulary (total number of unique tokens in the dataset).
  
  """
  
  def __init__(self,vocab_size):
    
    super().__init__()
    
    # Token embedding table: Maps each token to a vector representation
    self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
    
    # Positional embedding table: Encodes the position of each token in the sequence
    self.positional_embedding_table = nn.Embedding(block_size, n_emb)
    
    # Linear layer to map the embeddings to the vocabulary size (for predicting the next token)
    self.lm_head = nn.Linear(n_emb, vocab_size)
    
    # Multi-head attention layer to process the context of the tokens
    self.attention_heads = Multi_Attention_Head(4, n_emb // 4)  # 4 heads, with n_emb // 4 channels per head
    
    # Feedforward layer for further processing after attention
    self.ffd = feedforward(n_emb)


  def forward(self,idx,targets=None):
    """

      Performs the forward pass of the BiagramLanguageModel.

      Parameters:
      idx (torch.Tensor): The input tensor of token indices with shape (B, T), where B is the batch size 
                          and T is the sequence length.
      targets (torch.Tensor, optional): The target tensor for supervised learning (used for loss calculation). 
                                        Default is None, meaning no loss is computed.

      Returns:
      logits (torch.Tensor): The predicted logits (unnormalized scores) for each token in the vocabulary. 
                              Shape is (B, T, vocab_size).

    """
    # B: Batch size, T: Sequence length
    B, T = idx.shape
    
    # Retrieve token embeddings for each token in the input sequence
    token_embeddings = self.token_embedding_table(idx)  # Shape: (B, T, n_emb)
    
    # Retrieve positional embeddings for each position in the sequence
    positional_embeddings = self.positional_embedding_table(torch.arange(0, T, device=device))  # Shape: (T, n_emb)
    
    # Add token embeddings and positional embeddings to get the final embedding for each token
    x = token_embeddings + positional_embeddings
    
    # Apply multi-head attention to capture context dependencies
    x = self.attention_heads(x)
    
    # Apply the feedforward layer for further processing
    x = self.ffd(x)
    
    # Final linear layer to predict the next token probabilities
    logits = self.lm_head(x)  # Shape: (B, T, vocab_size)
    
    # To calculate the loss we use cross entropy however the format required by pytorch:
      # is (B*T,C) where all the elements are streatched into 1 dimensional sequence
    if targets is None:
      loss=None
    
    else:
      # The targets are also of shape : (B,T) and we want them to be : (B*T)
      B,T,C=logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T)
      loss=nn.functional.cross_entropy(logits,targets)
    
    return logits,loss

  def generate(self, idx, max_new_tokens):
      """
      Generates a sequence of new tokens from a given context using the trained language model.

      Parameters:
      idx (torch.Tensor): Input tensor of shape (B, T), where B is the batch size and T is the current sequence length.
                          This tensor represents the starting context of token indices.
      max_new_tokens (int): The number of new tokens to generate.

      Returns:
      torch.Tensor: A tensor of shape (B, T + max_new_tokens) containing the original context followed by the newly generated tokens.

      Description:
      This function iteratively generates `max_new_tokens` tokens by:
      1. Cropping the input to the last `block_size` tokens (to respect the model's context window).
      2. Feeding the cropped sequence into the model to get the logits.
      3. Selecting the logits corresponding to the last time step.
      4. Applying softmax to convert logits into probabilities.
      5. Sampling the next token index from the probability distribution.
      6. Appending the sampled token to the input sequence.
      
      The process is repeated until the desired number of tokens is generated.
      """
      for _ in range(max_new_tokens):
          # Crop idx to the last `block_size` tokens to maintain context size
          crop_idx = idx[:, -block_size:]

          # Get model predictions (logits), ignore loss since we're in inference
          logits,loss = self(crop_idx)

          # Focus on the last time step's logits (shape: B x vocab_size)
          logits = logits[:, -1, :]

          # Convert logits to probabilities
          probs = nn.functional.softmax(logits, dim=-1)  # Shape: (B, vocab_size)

          # Sample the next token from the probability distribution
          idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)

          # Append the predicted token to the input sequence
          idx = torch.cat((idx, idx_next), dim=1)  # Shape grows: (B, T+1)

      return idx

def evaluation(model: nn.Module, splits: list[str], eval_epochs: int) -> dict:
    """
    Evaluates the model's performance over multiple epochs for each specified data split.

    Parameters:
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    
    splits : list[str]
        A list of dataset split names to evaluate on (e.g., ["train", "val"]).
    
    eval_epochs : int
        The number of mini-batches to average the evaluation loss over for each split.

    Returns:
    -------
    dict
        A dictionary mapping each split name to its average loss, e.g., {"train": 1.25, "val": 1.53}.

    Description:
    -----------
    - Sets the model to evaluation mode (disables dropout, etc.).
    - Iterates through each specified split.
    - Computes and stores the average loss over `eval_epochs` sampled batches.
    - Restores the model to training mode before returning.
    """
    out = {}

    # Use inference mode for improved performance and memory savings
    with torch.inference_mode():
        model.eval()
        for split in splits:
            losses = torch.zeros(eval_epochs)
            for k in range(eval_epochs):
                x, y = get_batch(split)
                logits, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()  # Restore training mode

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