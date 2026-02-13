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
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class SelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_query: int = 128, 
                 n_heads: int = 8,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.n_heads = n_heads

        d_query = int(d_model / n_heads)

        self.norm = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_query)
        self.W_k = nn.Linear(d_model, d_query)
        self.W_v = nn.Linear(d_model, d_query)

        self.W_o = nn.Linear(d_model, d_model)

        self.scaling_factor = math.sqrt(d_query)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # normalize
        x_norm = self.norm(x)

        # duplicate tensor on heads dimension
        x_heads = x_norm.unsqueeze(1).repeat([1, self.n_heads, 1, 1]) # shape [batch_size, n_heads, seq_len, d_model]

        # calculate queries, keys, values
        q = self.W_q(x_heads) # shape [batch_size, n_heads, seq_len, d_query]
        k = self.W_k(x_heads) # shape [batch_size, n_heads, seq_len, d_query]
        v = self.W_v(x_heads) # shape [batch_size, n_heads, seq_len, d_query]

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
                 d_query: int = 128, 
                 n_heads: int = 8, 
                 n_layers: int = 4, 
                 d_up: int = 256,
                 device: torch.device = torch.device("cpu")
        ):
        self.n_vocab  = n_vocab
        self.d_model  = d_model
        self.d_query  = d_query
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
                                               (SelfAttention(config.d_model, config.d_query, config.n_heads, config.device),
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
            accum_steps: int = 2
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
            "accum_steps": self.accum_steps
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
            checkpoint["accum_steps"]
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
        return super().__len__() - self.start_idx



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
            outputs = outputs.reshape(-1, outputs.shape[-1])
            
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

    print(f"Starting training on epoch {start_epoch + 1}, batch {start_batch + 1}")

    compiled_model.train()

    batch_idx = start_batch

    for epoch in range(start_epoch, n_epochs):
        start_time = time.time()
        for idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = inputs[:,1:]
            outputs = compiled_model(inputs)[:,:-1,:]

            targets = targets.reshape(-1)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            
            loss = criterion(outputs, targets) / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % (accum_steps * 4) == 0:
                print(f"Epoch [{epoch + 1}].[{batch_idx + 1}] Loss: {loss * accum_steps}")

            if (batch_idx + 1) % (accum_steps * 64) == 0:
                print_avg_batch_time(start_time, accum_steps * 64)
                start_time = time.time()

                print_memory_usage()

                validate_model(compiled_model, device, criterion, validation_loader)

                checkpoint.batch = batch_idx
                save_checkpoint(model, optimizer, epoch, batch_idx + 1, checkpoint, checkpoint_path)
            batch_idx += 1

        batch_idx = 0
        save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint, checkpoint_path)
    epoch = 0
    save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint, checkpoint_path)



def generate_response(model, encoder: tiktoken.Encoding, device: torch.device, prompt: str):
    sequence = encoder.encode(prompt)
    with torch.no_grad():
        num_chars = 0
        end_sequence = False
        while end_sequence == False:
            input = torch.tensor(sequence, dtype=torch.int64).unsqueeze(0).to(device)
            output = model(input)
            output = output[0,-1,:].argmax().item()

            num_chars += 1
            if num_chars >= 1000:
                end_sequence = True
                
            if output == model.n_vocab - 1:
                end_sequence = True
            else:
                sequence = sequence + [output]
                
        response = encoder.decode(sequence)
        return response