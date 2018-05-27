
import numpy as np
import pickle as pkl
import time
import json
import random
import torch
from torch import nn
import editdistance
import os
from rnn.loader import make_loader, Preprocessor
from rnn.model import Seq2Seq
from rnn.model import LinearND 

np.seterr(divide='ignore') # masks log(0) errors

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

def main():
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

    print(model)
    
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


    # Testing goes here:


    #print("Testing RNN")
    #test_model = torch.load(os.path.join(config["save_path"], "best_0.8.16.0.5"))
    #test_ldr = make_loader(data_cfg["test_set"], preproc, opt_cfg["batch_size"])

    #_, test_wer = evaluate(test_model, test_ldr, preproc)

    #print("{:.2f}% WER (test)".format(test_wer * 100.))

if __name__ == __main__:
    main()
