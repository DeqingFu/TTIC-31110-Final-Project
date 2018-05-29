import numpy as np
import pickle as pkl
import time
import json
import random
import torch
from torch import nn
import editdistance
import os
import glob
from rnn.loader import make_loader, Preprocessor
from rnn.model import Seq2Seq
from rnn.model import LinearND 
from rnn.model import Attention
import matplotlib.pyplot as plt

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
        
    return np.nanmean(losses)

def evaluate(model, ldr, preproc, store_prediction=False, print_prediction=False):
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
    
    if store_prediction:
        with open("test_results.json", "w") as res:
            json.dump(results, res)

    if print_prediction:
        for (truth, pred) in results:
            print('True label:\n  ', end="")
            for char in truth:
                print(char, end=" ")
            print('\nPredicted labal:\n  ', end="")
            for char in pred:
                print(char, end=" ")
            print('')

    return np.nanmean(losses), compute_wer(results)

def main(training=False, testing=False):
    with open("rnn/config.json", "r") as fid:                                                                                                                                                                                                                                      
        config = json.load(fid)

    data_cfg = config["data"]
    model_cfg = config["model"]
    opt_cfg = config["optimizer"]
    start = time.time()
    preproc = Preprocessor(data_cfg["train_set"], start_and_end=data_cfg["start_and_end"])
    print("Preprocessing finished", time.time() - start, "seconds elapsed")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    if training:
        print("Training RNN")
        print("------------")
        start = time.time()
        train_ldr = make_loader(data_cfg["train_set"], preproc, opt_cfg["batch_size"])
        print("Train Loaded", time.time() - start, "seconds elapsed")   
        start = time.time()    
        dev_ldr = make_loader(data_cfg["dev_set"], preproc, opt_cfg["batch_size"])
        print("Dev Loaded", time.time() - start, "seconds elapsed")
        print("All Data Loaded")

        attention = Attention(model_cfg["encoder"]["hidden_size"], model_cfg["decoder"]["hidden_size"], 64)
        model = Seq2Seq(preproc.input_dim, preproc.vocab_size, attention, model_cfg)
        model = model.cuda() if use_cuda else model.cpu()

        optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg["learning_rate"], momentum=opt_cfg["momentum"])
        mslst = [int(y) for y in [25 * x for x in range(1,20)] if y < opt_cfg["max_epochs"]]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mslst, gamma=0.1)
        
        log="epoch {:4} | train_loss={:6.2f}, dev_loss={:6.2f} with {:6.2f}% WER ({:6.2f}s elapsed)"
        #log="epoch {:4} | train_loss={:6.2f}, dev_loss={:6.2f} ({:6.2f}s elapsed)"
        losses = []
        weres = []
        eps = list(range(opt_cfg["max_epochs"]))
        best_so_far = float("inf")
        for ep in range(opt_cfg["max_epochs"]):
            start = time.time()
            scheduler.step()

            train_loss = train(model, optimizer, train_ldr)    
            dev_loss, dev_wer = evaluate(model, dev_ldr, preproc)
            losses.append(dev_loss)
            weres.append(dev_wer)
            #dev_loss = evaluate(model, dev_ldr, preproc)        
            
            print(log.format(ep + 1, train_loss, dev_loss, dev_wer * 100., time.time() - start))
            for param_group in optimizer.param_groups:
                print('...learning rate: ' + str(param_group['lr']))
            #print(log.format(ep + 1, train_loss, dev_loss, time.time() - start))
            
            torch.save(model, os.path.join(config["save_path"], str(ep)))   
            if dev_loss < best_so_far:
                best_so_far = dev_wer
                torch.save(model, os.path.join(config["save_path"], "best"))
        plt.plot(eps, losses)
        plt.plot(eps, weres)
        plt.show()
    if training and testing:
        print("")
    
    # Testing goes here:
    if testing:
        print("Testing RNN")
        print("-------------")

        test_model = torch.load(os.path.join(config["save_path"], "best"))
        start = time.time()
        test_ldr = make_loader(data_cfg["test_set"], preproc, opt_cfg["batch_size"])
        print("Test Loaded", time.time() - start, "seconds elapsed")

        _, test_wer = evaluate(test_model, test_ldr, preproc, True, True)
        #test_loss = evaluate(test_model, test_ldr, preproc)

        print("{:.2f}% WER (test)".format(test_wer * 100.))
        #print("Test loss", test_loss)

    print("----")
    print("done")
    
if __name__ == '__main__':
    main(training=False, testing=True)
