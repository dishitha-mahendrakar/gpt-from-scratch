import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for productions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ability to run on a gpu if you have it
eval_iters = 200
n_embd = 32 # a suggestion from github copilot
# ------------

# reproducibility
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# here are all the unqiue characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch : i for i, ch in enumerate(chars) }
itos = { i : ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder : take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder : take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data)) #  first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i + 1:i +block_size + 1]for i in ix])
    x, y = x.to(device), y.to(device) # when we load the data, we move that to device
    return x, y

@torch.no_grad() # context manager
# also a good practice to tell PyTorch to tell when we don't intend
# to do backpropagation

# just telling PyTorch that everything that happens inside 
# estimate_loss function, we will not call .backward() on
# and so pytorch can be a lot more memory efficient with its memory use
# because it does not have to store all the intermediate variables
# beacause we are never going to call backward

# it average up the loss over multiple batches
def estimate_loss():
    out = {}
    model.eval() # setting model to evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # average loss for both splits
    model.train() # resetting it back to training phase
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # here is the nn embedding table it has got a .weight inside which stores the lookup table which will be moved to the gpu so that all the calcuations happen on the gpu and that can be a lot faster
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # so each position from 0 to block_size -1 will also get its own embedding vector
        self.lm_head = nn.Linear(n_embd, vocab_size)

# this just creates one spurious level of interaction
# through a linear layer

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        # the positional embedding
        # torch.arange() 
        # integers from 0 to T-1 all of which get embedded through the table
        # to create (T, C)

        # lets just say that an emb = C

        x = tok_emb + pos_emb # (B,T,vocab_size)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # now the broadcasting note will work out
        # (B,T,C) + (T,C)
        # this gets right aligned
        # a new dimension of 1 gets added 
        # and it gets broadcasted acorss batch

        # so at this point x holds
        # not just the token identities
        # but the positions at which these tokens occur
        # this is currently not that useful because
        # of course we just have a simple bigram model
        # so it does not matter if you are on the fifth position or 2nd posn or wherever
        # it's all tranlation invariant at this stage
        # so, this information currently wouldn't help
        # but as we work on the self attention block
        # we'll see that this starts to matter 


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) #(B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        return idx

model = BigramLanguageModel()
m = model.to(device) # move the model parameters to device

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate from the model
# need to make sure that we create the context on the device
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

# this model behaves the same for training and evaluation phase
# because the only thing in the model is nn.Embedding
# we have no dropout layers , no batchNorm layers etc
# but it is a good practice to think through what mode your NN is in
# because some layers will have differnt behaviour during training time or inference time

# vocab_size
# no need to pass as a parameter to constructor
# beause it is already defined up top as a global variable
# there is no need to pass this stuff around

# create a level of interaction here
# where we dont directly go to the embedding
# for the logits
# but instead we go thorugh this intermediate 
# phase because we are going to start making it bigger

# let us introduce a new variable n_embd short for
# number of embedding dimensions

# it is not going to give us logits directly
# it is going to give us token embeddings 'tok_emb'
# to go from tok_emb --> logits , we're gonna need
# a linear layer

# 11:45pm
# def forward currently looks kinda spurious but we are going to build on top of it

# so far we have taken these indices ( idx in forward()) and we have encoded them
# based on the identity of the tokens inside idx


# the next thing that people very often do is that
# not just encoding the identity of these indcies but alos their position
# so, we're going to have a second position embedding table here