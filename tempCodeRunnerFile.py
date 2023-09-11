class FeedForward(nn.Module):
    """a simple linear layer followed by a ReLU non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
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

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # here is the nn embedding table it has got a .weight inside which stores the lookup table which will be moved to the gpu so that all the calcuations happen on the gpu and that can be a lot faster
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # so each position from 0 to block_size -1 will also get its own embedding vector
        
        # creating a head in the constructor
        # callling it the self attention head
        # head_size will be kept the same = n_embd
        self.sa_heads = MultiHeadAttention(4,n_embd//4) #i.e 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
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
        # feed it into the self attention head
        x = self.sa_heads(x)  # apply one head of self-attention. (B,T,C)
        # ffwd called sequentially right after self-attention
        x = self.ffwd(x) # (B,T,C)
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
'''