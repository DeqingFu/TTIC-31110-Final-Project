
# coding: utf-8

# In[3]:


import numpy as np
import pickle as pkl
import time
import json
import random
import torch
from torch import nn
import editdistance
import os
np.seterr(divide='ignore') # masks log(0) errors

from rnn.loader import make_loader, Preprocessor
from rnn.model import Seq2Seq
from rnn.model import LinearND #Hint: this is useful when defining the modified attention mechanism


# In[5]:


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        """
        Initialize Attention.
        ----
        enc_dim: encoder hidden state dimension
        dec_dim: decoder hidden state dimension
        attn_dim: attention feature dimension
        """
        super(Attention, self).__init__()
        if enc_dim == dec_dim and attn_dim is None:
            self.use_default = True
        elif attn_dim is not None:
            self.use_default = False
            self.attn_dim = attn_dim
            self.enc_dim = enc_dim
            self.dec_dim = dec_dim
            self.v = LinearND(self.attn_dim, 1, bias=False)
            self.W1 = LinearND(self.enc_dim, self.attn_dim, bias=False)
            self.W2 = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        else:
            raise ValueError("invalid args (enc_dim={}, dec_dim={}, attn_dim={})".format(enc_dim, dec_dim, attn_dim))

    def forward(self, eh, dhx, ax=None):
        """
        Forward Attention method.
        ----
        eh (FloatTensor): the encoder hidden state with
            shape (batch size, time, hidden dimension).
        dhx (FloatTensor): one time step of the decoder hidden
            state with shape (batch size, hidden dimension).
        ax (FloatTensor): one time step of the attention vector.
        ----
        Returns the context vectors (sx) and the corresponding attention alignment (ax)
        """
        
        if self.use_default:
            # Compute inner product of decoder slice with every encoder slice
            pax = torch.sum(eh * dhx, dim=2)
            ax = nn.functional.softmax(pax, dim=1)
            sx = torch.sum(eh * ax.unsqueeze(2), dim=1, keepdim=True)
        else:
            # Alternative attention mechanism implemented
            pax = torch.sum(self.v(nn.functional.tanh(self.W1(eh) + self.W2(dhx))), dim=2)
            ax = nn.functional.softmax(pax, dim=1)
            sx = torch.sum(eh * ax.unsqueeze(2), dim=1, keepdim=True)
        return sx, ax


# In[6]:


def compute_wer(results):
    """
    Compute the word-error-rate (WER).
    """
    dist = 0.
    for label, pred in results:
        dist += editdistance.eval(label, pred)
    total = sum(len(label) for label, _ in results)
    return dist / total

def train(model, optimizer, ldr):
    """
    Train the model for an epoch (one pass over the training data)
    ----
    model: Seq2Seq model instance
    optimizer: torch.nn optimizer instance
    ldr: data loader instance
    ----
    Returns the average loss over an epoch
    """
    model.train()
    model.scheduled_sampling = model.sample_prob != 0
    
    losses = []
    
    for ii, (inputs, labels) in enumerate(ldr):
        optimizer.zero_grad()
        x, y = model.collate(inputs, labels)
        loss = model.loss(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss.data.item())
        
    return np.mean(losses)

def evaluate(model, ldr, preproc):
    """
    Evaluate the model (on either dev or test).
    ----
    model: Seq2Seq model instance
    ldr: data loader instance
    preproc: preprocessor instance
    ----
    Returns the average loss and wer on a given dataset
    """
    model.eval()
    model.scheduled_sampling = False
    
    losses, hyps, refs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in ldr:
            x, y = model.collate(inputs, labels)
            # get loss
            loss = model.loss(x, y)
            losses.append(loss.data.item())
            # get predictions
            pred = model.infer(x, y)
            hyps.extend(pred)
            refs.extend(labels)

    results = [(preproc.decode(r), preproc.decode(h)) for r, h in zip(refs, hyps)]
    
    return np.mean(losses), compute_wer(results)


# In[42]:


"""
Use the development set to tune your model.
------
With the default config, can get <10% dev WER within 15 epochs.
"""

with open("rnn/config.json", "r") as fid:                                                                                                                                                                                                                                      
    config = json.load(fid)

#random.seed(config["seed"])
#np.random.seed(config["seed"])
#torch.manual_seed(config["seed"])
#
#use_cuda = torch.cuda.is_available()
#if use_cuda:
#    torch.backends.cudnn.deterministic = True

print("Printing RNN")
data_cfg = config["data"]
model_cfg = config["model"]
opt_cfg = config["optimizer"]

preproc = Preprocessor(data_cfg["train_set"], start_and_end=data_cfg["start_and_end"])

#train_ldr = make_loader(data_cfg["train_set"], preproc, opt_cfg["batch_size"])
#dev_ldr = make_loader(data_cfg["dev_set"], preproc, opt_cfg["batch_size"])

attention = Attention(model_cfg["encoder"]["hidden_size"], model_cfg["decoder"]["hidden_size"])
model = Seq2Seq(preproc.input_dim, preproc.vocab_size, attention, model_cfg)
#model = model.cuda() if use_cuda else model.cpu()

optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg["learning_rate"], momentum=opt_cfg["momentum"])

#log="epoch {:4} | train_loss={:6.2f}, dev_loss={:6.2f} with {:6.2f}% WER ({:6.2f}s elapsed)"

#best_so_far = float("inf")
#for ep in range(opt_cfg["max_epochs"]):
#    start = time.time()
#    
#    train_loss = train(model, optimizer, train_ldr)    
#    dev_loss, dev_wer = evaluate(model, dev_ldr, preproc)
#    
#    print(log.format(ep + 1, train_loss, dev_loss, dev_wer * 100., time.time() - start))
#    
#    torch.save(model, os.path.join(config["save_path"], str(ep)))
#    
#    if dev_wer < best_so_far:
#        best_so_far = dev_wer
#        torch.save(model, os.path.join(config["save_path"], "best"))


# In[25]:


#print("Testing RNN")
#test_model = torch.load(os.path.join(config["save_path"], "best_0.8.16.0.5"))
#test_ldr = make_loader(data_cfg["test_set"], preproc, opt_cfg["batch_size"])

#_, test_wer = evaluate(test_model, test_ldr, preproc)

#print("{:.2f}% WER (test)".format(test_wer * 100.))
print(model)
