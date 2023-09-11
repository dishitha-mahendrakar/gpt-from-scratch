"""
Dropout comes from the paper from 2014,
Dropout: A simple way to prevent neural networks from overfitting

dropout takes your NN
and it randomly
every forward backward pass
shuts off some subset of neurons

randomly drops them to zero and
trains without them

and what this does effectively is 
because the mask of what is being dropped out
has changed 
every single forward backward pass
it ends up kind of training an ensemble of sub networks

and then at test time
everything is fully enabled
and all of those sub networks are merged
into a single ensemble if you can think about it that way

i would read the whole paper to get the full detail
for now we're just going to stay on the level of 
this is a regularization technique and
we added this because we are going scale up the model 
quite a bit and i was concerned about overfitting
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for productions?
max_iters = 5000
# also increased the num of iters 
# because the learning rate is lower
eval_interval = 500
# decreased the learning rate after v2
# because the self attention cannot tolerate very very high learning rates
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ability to run on a gpu if you have it
eval_iters = 200
n_embd = 384 # a suggestion from github copilot
n_head = 6 # which means that every head is 384/6 = 64 dim as a standard
n_layer = 6
dropout = 0.2 
# so every forward backward pass 20% of all of 
# intermediate calculations are disable and dropped to zero


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

class Head(nn.Module): # head module
    """one head of self-attention""" # <- implements
    def __init__(self, head_size): # you give it a head_size
        super().__init__()
        # it creates the key, query and value linear layers
        # typically people don't use biases in these
        # these are the linear projections taht we are goign to apply
        # to all of our nodes
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        # we are creating a tril variable
        # so instead of PyTorch naming convensions
        # it is called a buffer
        # tril is not a parameter of the module
        # you have to call it
        # you have to assign it to the module using a register buffer
        # so that creates the tril - the lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    # and when we are given the input x
    # this should look very familiar now
    def forward(self, x):
        B,T,C = x.shape
        # we calcluate the keys
        # the queries
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores ("affinities") inside wei
        # we normalize it so we are using scaled attention * C ** -0.5
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        # then we make sure that future does not communicate with the past 
        # so this makes it a decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,C)

        # then softmax 
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # we can randomly prevent some of the nodes from 
        # communicating
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        # output
        out = wei @ v #(B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    '''in pytorch we can do this by simply creating 
    multiple heads 
    '''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        '''
        initialize projection
        '''
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    '''run all of them in parallel into a list
    and simply concatenate all of the outputs and 
    we're concatenating over the channel dimension
    '''
    def forward(self, x):
        # output of self attention itself
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        # apply the projection
        # projection is the linear transformation of the 
        # outcome of this - torch.cat([h(x) for h in self.heads], dim = -1)
        # layer.
        out = self.proj(out) # the projection back into the residual pathway
        return out

'''
we don't just have a single attention
that has a head_size  = 32

instead of having one communication channel
we now have 4 communication channels in parallel
and each one of these communcation channels 
typically will be smaller correspondingly

because we have 4 communication channels 
we want 8 dimensional self-attention

and so from each communicaiton channel
we are going to gatehr 8 dimensional vectors
and then we have 4 of them 
which concatenates to give us 32 which is the original n_embd

this is kind of like a group convolution
because basically instead of having one large convolution
we do convolution in groups
and that's multi-headed self attention
'''   

class FeedForward(nn.Module):
    """a simple linear layer followed by a ReLU non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4  * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),# the projection layer going back into the residual pathway
            # dropout is something that you can add
            # right before the connection back into the
            # original pathway
            nn.Dropout(dropout),
        )

    '''
    this is on a per token level
    all the tokens do this independently
    the self attention is the communication
    now that they have gathered all teh data
    they need to think on that data individually
    taht's what feed forward is doing
    '''
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # head_size becomes 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        # and we tell it what is the embedding dimension
        self.ln2 = nn.LayerNorm(n_embd)
        # here the layer norms are applied immediately on x
        # the layer norms are applied immediately on x
        
        # size of layerNorm is 32
        # so when the layerNorm is normalizing our features
        # it is the normalization here happens
        # the mean and the variance are taken over 32 nums
        # so the batch and the time 
        # both of them act as batch dimensions
        # this is kind of like a per token transformation
        # that just normalizes the features and makes them
        # unit mean
        # unit gaussian
        # at initialization
        # but ofcourse because these layer norms inside it
        # have these gamma and beta trainable parameters
        # the layer norm will eventually create outputs
        # that might not be unit gaussian but the 
        # optimization will determine that
    ''' 
    this is how we want to intersprese them
    '''
    def forward(self, x):
        '''
        we fork off and do some communication 
        and come back
        '''
        # layerNorm applied before it goes into self-attention        
        x = x + self.sa(self.ln1(x))
        '''
        we fork off and we do some computation
        and come back
        '''
        # layerNorm applied before it goes into feed-forward        
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # here is the nn embedding table it has got a .weight inside which stores the lookup table which will be moved to the gpu so that all the calcuations happen on the gpu and that can be a lot faster
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # so each position from 0 to block_size -1 will also get its own embedding vector
        '''
        sequential application of Block-s
        so that we are interspersing 
        communication feed forward many many times
        '''
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


# this just creates one spurious level of interaction
# through a linear layer

    '''
    before we had the multiheaded self attention
    that did the communication
    but we went way to fast to calculate the logits
    so, the tokens looked at each other but 
    didn't really have a lot of time to  think on what they 
    found from the other tokens'''

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

        # encoding of information using the token embdgs, and pos embdgs
        x = tok_emb + pos_emb # (B,T,vocab_size)
        ''' then finally we decode '''
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)
        # output is going into the decoder language modelling head
        # and create the logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        # this was the simplest way to plug in a self-attention 
        # componnet into our network right now

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

    # in generate we have to make sure that 
    # our idx that we feed into the model
    # because now we are using posotioanl embeddings
    # we can never have more than block_size coming in 
    # because if idx is more than block_size
    # then our position embedddings table is going to run out
    # of scope
    # because it only has embedding for upto block size
    # therefore we add code
    # to crop the context
    # that we are going to feed into self
    # so that we never pass in more than block_Size elements

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
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
    if iter == 4999:
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
'''
this model behaves the same for training and evaluation phase
because the only thing in the model is nn.Embedding
we have no dropout layers , no batchNorm layers etc
but it is a good practice to think through what mode your NN is in
because some layers will have differnt behaviour during training time or inference time

vocab_size
no need to pass as a parameter to constructor
beause it is already defined up top as a global variable
there is no need to pass this stuff around

create a level of interaction here
where we dont directly go to the embedding
for the logits
but instead we go thorugh this intermediate 
phase because we are going to start making it bigger

let us introduce a new variable n_embd short for
number of embedding dimensions

it is not going to give us logits directly
it is going to give us token embeddings 'tok_emb'
to go from tok_emb --> logits , we're gonna need
a linear layer

11:45pm
def forward currently looks kinda spurious but we are going to build on top of it

so far we have taken these indices ( idx in forward()) and we have encoded them
based on the identity of the tokens inside idx


the next thing that people very often do is that
not just encoding the identity of these indcies but alos their position
so, we're going to have a second position embedding table here


previously
step 2700: train loss 2.5006, val loss 2.5117

after self attenion 
step 2500: train loss 2.4272, val loss 2.4435
step 4500: train loss 2.3980, val loss 2.4084

the text is still not amazing
clearly teh self attention head is doing some useful communication
but we still have a long way to go

multihead attention
applying mulitple attention in parallel and concatenating the results

after multi-head self-attention:
step 4500: train loss 2.2748, val loss 2.2858

the generation is still not amazing
but clearly the validation loss is improving

it helps to have multiple communicaton channels
because obviously these tokens have a lot to talk about

they want to find the consonants 
they want to find the vowels just from certain positions
they want to find any kinds of different things

and so it helps to create multiple indepdentent channels of 
communication
gather lots of differet types of data
and then decode the output

after adding computation:
step 4500: train loss 2.2282, val loss 2.2409

the output still looks terrible 
but we have imporved the situation  

we are going to now intersperse the
communication with the computation

and that is what the Transformer does
when it has blocks that communication and then compute
and it groups them
and replicates them

after transformer block:
step 4500: train loss 2.3178, val loss 2.3336

not very good result 
the reason -
we are starting to actually get like a pretty deep NN
and deep neural nets suffer from
optimization issues

that is where we're kind of like slightly starting
to run into

there are two optimizations 
that dramatically help 
with the depth of these networks
and make sure that 
the networks remain optimizable
let's talk about the first one

skip connections / residual connections

Paper : deep residual learning for Image Recognition,2015
that introduced the concept

after training : residual pathway - 
step 4999: train loss 1.9995, val loss 2.0822
we also see that
the network is starting to get big enough

our train loss is getting ahead of val loss

we are starting to see a little bit of overfitting

our generations here - 
The sopalast tuse harith pon, but my to the sArvore akmimpter?'dl as masted provend, his thou rold the this brighile'ly as down. E. Dant my nep as but as stles, bum,
The. 
Wat, you your bur.' slastand, fouge.

QUERED VING RIS:
The'll agl the fapcion it sheentroben losed on fuld aplort a let the reavings shous the deciiverving is my lord tay will I los that negition comence's he it as me pread may mallifes I suchate RICLick leaves and willd purwered Joustars is nold.

VARK:
It spartiling.

NORLAN

are still not amazing but
we can see little formation of sentences like -
 "is my", "to the"

like this starts to almost look like english

we're starting to really get there!

after layerNorm:
step 4999: train loss 1.9795, val loss 2.0595

Times pall'd trake:
Ait speald I and to the sAnvoust king them'dlses mastede Bromey, his thou rolde good inf.

MEENENE:
Nou the and in this nep as but as wills, brothis,
Thell, you your bur. I was the'd theeeef osshe doo lood the blood my a happir your sheeet aben losed on of on plort a let the rackings shome the deceivery manish to lord at will I los that neg-tidy crike, of he it gramedpsed, and your I connour and Roweiry leam;
And my dade up now, Jule ars is nold.

VARd to the howll'd it the b


'''