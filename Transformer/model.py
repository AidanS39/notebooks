import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import v2
from datasets import load_dataset
import tiktoken
import time
import math
import os

from util import EpochLog
from util import TrainLog

# taken from https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, d_model]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class SelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.n_heads = n_heads

        self.d_query = int(d_model / n_heads)

        self.norm = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.scaling_factor = math.sqrt(self.d_query)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # normalize
        x_norm = self.norm(x) # shape [batch_size, seq_len, d_model]

        # calculate queries, keys, values
        q = self.W_q(x_norm) # shape [batch_size, seq_len, d_model]
        k = self.W_k(x_norm) # shape [batch_size, seq_len, d_model]
        v = self.W_v(x_norm) # shape [batch_size, seq_len, d_model]

        # reshape for heads
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d_query) # shape [batch_size, seq_len, n_heads, d_model]
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d_query) # shape [batch_size, seq_len, n_heads, d_model]
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d_query) # shape [batch_size, seq_len, n_heads, d_model]

        q = torch.transpose(q, -3, -2) # shape [batch_size, n_heads, seq_len, d_model]
        k = torch.transpose(k, -3, -2) # shape [batch_size, n_heads, seq_len, d_model]
        v = torch.transpose(v, -3, -2) # shape [batch_size, n_heads, seq_len, d_model]

        # form attention pattern
        attention_pattern = torch.matmul(q, torch.transpose(k, -2, -1)) / self.scaling_factor # shape [batch_size, n_heads, seq_len, seq_len]
        
        # create and apply mask
        seq_len = attention_pattern.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
        attention_pattern = torch.masked_fill(attention_pattern, mask, float("-inf"))

        # apply softmax
        attention_pattern = self.softmax(attention_pattern)
        attention_vectors = torch.matmul(attention_pattern, v) # shape [batch_size, n_heads, seq_len, d_query]

        # concat heads
        output = torch.transpose(attention_vectors, -2, -1) # shape [batch_size, n_heads, d_query, seq_len]
        output = torch.flatten(output, start_dim=-3, end_dim=-2) # shape [batch_size, d_model, seq_len]
        output = torch.transpose(output, -2, -1) # shape [batch_size, seq_len, d_model]

        # final output projection
        output = self.W_o(output) # shape [batch_size, seq_len, d_model]

        # residual connections
        output = output + x

        return output



class MultilayerPerceptron(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_up: int = 256):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.up = nn.Linear(d_model, d_up)
        self.relu = nn.ReLU()
        self.down = nn.Linear(d_up, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_norm = self.norm(x)

        output = self.up(x_norm)
        output = self.relu(output)
        output = self.down(output)

        output = output + x

        return output

class TransformerConfig():
    def __init__(self, 
                 n_vocab: int, 
                 d_model: int = 128,
                 n_heads: int = 8, 
                 n_layers: int = 4, 
                 d_up: int = 256,
                 device: torch.device = torch.device("cpu")
        ):
        self.n_vocab  = n_vocab
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.d_up     = d_up
        self.device   = device

class Transformer(nn.Module):
    def __init__(self, 
                 config: TransformerConfig):
        super().__init__()

        self.n_vocab = config.n_vocab

        self.embedding = nn.Embedding(config.n_vocab, config.d_model)
        self.pe = PositionalEncoding(config.d_model, max_len=50000)

        self.attention_layers = nn.ModuleList([layer for _ in range(config.n_layers) for layer in 
                                               (SelfAttention(config.d_model, config.n_heads, config.device),
                                                MultilayerPerceptron(config.d_model, config.d_up))])

        self.unembedding = nn.Linear(config.d_model, config.n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pe(x)

        for layer in self.attention_layers:
            x = layer(x)

        x = self.unembedding(x)
        
        return x



class TransformerCheckpoint():
    def __init__(
            self, 
            model_state, 
            optim_state, 
            config: TransformerConfig, 
            rng_state: torch.Tensor, 
            encoder, 
            epoch: int = 0, 
            batch: int = 0, 
            n_epochs: int = 3, 
            accum_steps: int = 2,
            train_log: TrainLog = None
        ):
        self.model_state = model_state
        self.optim_state = optim_state
        self.config      = config
        self.rng_state   = rng_state
        self.encoder     = encoder
        self.epoch       = epoch
        self.batch       = batch
        self.n_epochs    = n_epochs
        self.accum_steps = accum_steps
    
        if train_log == None:
            train_log = TrainLog()
        self.train_log = train_log

    
    def save(self, file_path):
        print(f"Saving model checkpoint at {file_path}.... DO NOT EXIT")
        torch.save({
            "model_state": self.model_state,
            "optim_state": self.optim_state,
            "config"     : self.config,
            "rng_state"  : self.rng_state,
            "encoder"    : self.encoder,
            "epoch"      : self.epoch,
            "batch"      : self.batch,
            "n_epochs"   : self.n_epochs,
            "accum_steps": self.accum_steps,
            "train_log"  : self.train_log
        }, file_path)
        print(f"Model checkpoint saved at {file_path} at epoch {self.epoch} batch {self.batch}")
    
    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        return TransformerCheckpoint(
            checkpoint["model_state"], 
            checkpoint["optim_state"], 
            checkpoint["config"], 
            checkpoint["rng_state"],
            checkpoint["encoder"],
            checkpoint["epoch"], 
            checkpoint["batch"],
            checkpoint["n_epochs"],
            checkpoint["accum_steps"],
            checkpoint["train_log"]
        )



class CheckpointRandomSampler(torch.utils.data.RandomSampler):
    def __init__(self, data_source, batch_size, checkpoint: TransformerCheckpoint = None):
        super().__init__(data_source)
        self.start_batch_idx = 0 if checkpoint == None else checkpoint.batch
        self.batch_size = batch_size
    def __iter__(self):
        idxs = list(super().__iter__())
        for idx in idxs[(self.start_batch_idx*self.batch_size):]:
            yield idx
        return
    def __len__(self):
        return super().__len__() - self.start_batch_idx



def print_memory_usage() -> None:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"USAGE: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB, Peak: {peak:.2f}GB")



def print_avg_batch_time(start_time, num_batches) -> None:
    elapsed_time = time.time() - start_time
    print(f"TIME: {elapsed_time / (num_batches)} seconds per batch")



def save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint: TransformerCheckpoint, checkpoint_path):
    checkpoint.model_state = model.state_dict()
    checkpoint.optim_state = optimizer.state_dict()
    checkpoint.batch = batch_idx
    checkpoint.epoch = epoch
    checkpoint.save(checkpoint_path)



def validate_model(model, 
                   device, 
                   criterion, 
                   test_loader):
    with torch.no_grad():
        avg_loss = 0
        for idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = inputs[:,1:]
            outputs = model(inputs)[:,:-1,:]

            targets = targets.reshape(-1)
            outputs = outputs.reshape(-1, outputs.shape[-1]).float()
            
            loss = criterion(outputs, targets)

            avg_loss += loss

        avg_loss /= len(test_loader)
        print(f"VALIDATE: Average Loss: {avg_loss}")



def train_model(model: nn.Module, 
                optimizer, 
                criterion, 
                device, 
                train_loader, 
                validation_loader,
                checkpoint: TransformerCheckpoint,
                checkpoint_path: str = "checkpoint.pt"):
    
    start_epoch = 0
    start_batch = 0

    if checkpoint != None:
        model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optim_state)
        start_epoch = checkpoint.epoch
        start_batch = checkpoint.batch
        n_epochs    = checkpoint.n_epochs
        accum_steps = checkpoint.accum_steps
        torch.set_rng_state(checkpoint.rng_state)
    else:
        raise Exception("Error: checkpoint passed as None")
        
    compiled_model = torch.compile(model, options={"split_reductions": False})

    print(f"Starting training on epoch {start_epoch}, batch {start_batch}")
    
    checkpoint.train_log.start_timer()

    compiled_model.train()

    batch_idx = start_batch

    for epoch in range(start_epoch, n_epochs):
        epoch_log = EpochLog(epoch)
        start_time = time.time()
        for idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = inputs[:,1:] # shape [batch_size, seq_len - 1]
            targets = targets.reshape(-1) # shape [batch_size * (seq_len - 1)]
            
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                outputs = compiled_model(inputs)[:,:-1,:] # shape [batch_size, seq_len - 1, n_vocab]
                outputs = outputs.reshape(-1, outputs.shape[-1]) # shape [batch_size * (seq_len - 1), n_vocab]
                
                loss = criterion(outputs, targets) / accum_steps

            loss.backward()

            batch_idx += 1

            if (batch_idx) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx) % (accum_steps * 4) == 0:
                epoch_log.add_batch_log(batch_idx - 1, loss.item() * accum_steps)
                print(f"Epoch [{epoch}].[{batch_idx - 1}] Loss: {loss * accum_steps}")

            if (batch_idx) % (accum_steps * 64) == 0:
                print_avg_batch_time(start_time, accum_steps * 64)
                start_time = time.time()

                print_memory_usage()

                validate_model(compiled_model, device, criterion, validation_loader)

                save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint, checkpoint_path)

        batch_idx = 0
        checkpoint.train_log.add_epoch_log(epoch_log)
        save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint, checkpoint_path)
        torch.cuda.empty_cache()
    epoch = n_epochs - 1
    checkpoint.train_log.stop_timer()
    save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint, checkpoint_path)



def generate_response(model, encoder: tiktoken.Encoding, device: torch.device, prompt: str, temp: float = 1, k: int = 5):
    sequence = encoder.encode(prompt)

    for token in sequence:
        print(encoder.decode([token]), end="", flush=True)
        time.sleep(0.05)
    with torch.no_grad():
        num_chars = 0
        end_sequence = False
        while end_sequence == False:
            input = torch.tensor(sequence, dtype=torch.int64).unsqueeze(0).to(device)
            logits = model(input)[0,-1,:]

            # sample token from softmax of top k
            logits = torch.topk(logits, k=k, dim=-1)
            values = logits.values / temp
            values = values.softmax(dim=-1, dtype=torch.float)
            output_token = logits.indices[int(torch.multinomial(values, num_samples=1, replacement=True))]


            num_chars += 1
            if num_chars >= 1000:
                end_sequence = True
                
            if output_token == model.n_vocab - 2:
                end_sequence = True
            else:
                sequence = sequence + [output_token]
                print(encoder.decode([output_token]), end="", flush=True)
                time.sleep(0.05)
        response = encoder.decode(sequence)
        return response