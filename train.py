import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch._inductor.config as inductor_config
from datasets import load_dataset
import tiktoken
import os
import argparse

from model import Transformer
from model import TransformerCheckpoint
from model import TransformerConfig
from model import CheckpointRandomSampler
from model import train_model

from utils import get_device

def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--help', action='help', help='help menu')

    parser.add_argument("-m", "--d_model", type=int, default=1024, help="model dimension")
    parser.add_argument("-h", "--n_heads", type=int, default=8, help="number of heads")
    parser.add_argument("-l", "--n_layers", type=int, default=8, help="number of layers")
    parser.add_argument("-u", "--d_up", type=int, default=2048, help="up dimension")
    parser.add_argument("-e", "--n_epochs", type=int, default=5, help="number of epochs")

    args = parser.parse_args()

    print(f"D_MODEL {args.d_model}")
    print(f"N_HEADS {args.n_heads}")
    print(f"N_LAYERS {args.n_layers}")
    print(f"D_UP {args.d_up}")
    print(f"N_EPOCHS {args.n_epochs}")

    return args

def main():

    # get user defined model parameters
    args = get_arguments()

    # set environment parameters
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    inductor_config.split_reductions = False

    # get compute device
    device = get_device()

    # load and tokenize dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train+validation")

    encoder = tiktoken.get_encoding("cl100k_base")

    def tokenize(sequence, encoder):
        # NOTE: <EOS> token is encoder.n_vocab
        sequence["text"] = torch.tensor(encoder.encode(sequence["text"]) + [encoder.n_vocab], dtype=torch.int64)
        return sequence
    
    tokenized_dataset = dataset.map(tokenize, num_proc=16, fn_kwargs={"encoder": encoder}).with_format("torch")
    
    # split dataset into 80% train, 20% test
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)
    train = tokenized_dataset["train"]["text"]

    # further split test set into 99.9% test, 0.01% validation
    test_set = tokenized_dataset["test"].train_test_split(test_size=0.001, shuffle=True)
    test = test_set["train"]["text"]
    validation = test_set["test"]["text"]
    
    # set model and training parameters
    model_config = TransformerConfig(
        n_vocab=encoder.n_vocab + 2, # NOTE: add 2, 1 for <EOS> token and 1 for padding token
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_up=args.d_up,
        device=device
    )

    n_epochs      = args.n_epochs
    batch_size    = 8
    accum_steps   = 4
    learning_rate = 0.0001
    padding_value = model_config.n_vocab - 1 # ignore padding value during loss
    checkpoint_path = f"./models/model_{model_config.d_model}_{model_config.n_heads}_{model_config.n_layers}_{model_config.d_up}.pt"

    # initialize model with configuration
    model = Transformer(model_config).to(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initialize checkpoint with user defined parameters
    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = TransformerCheckpoint.load(checkpoint_path)
    else:
        checkpoint = TransformerCheckpoint(
            model.state_dict(),
            optimizer.state_dict(),
            model_config,
            torch.get_rng_state(),
            encoder,
            n_epochs=n_epochs,
            accum_steps=accum_steps
        )

    # define data loaders and sampler
    sampler = CheckpointRandomSampler(train, batch_size, checkpoint)

    def collate_fn_padding(batch):
        batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
        return batch

    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_padding)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padding)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padding)

    # start training loop
    train_model(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device,
        train_loader=train_loader, 
        validation_loader=validation_loader,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    main()