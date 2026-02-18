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

from model import Transformer
from model import TransformerCheckpoint
from model import TransformerConfig
from model import CheckpointRandomSampler
from model import train_model

def main():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    inductor_config.split_reductions = False

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    dataset = load_dataset("roneneldan/TinyStories", split="train+validation")

    encoder = tiktoken.get_encoding("cl100k_base")

    def tokenize(sequence, encoder):
        # NOTE: <EOS> token is encoder.n_vocab
        sequence["text"] = torch.tensor(encoder.encode(sequence["text"]) + [encoder.n_vocab], dtype=torch.int64)
        return sequence

    tokenized_dataset = dataset.map(tokenize, num_proc=8, fn_kwargs={"encoder": encoder}).with_format("torch")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)

    train = tokenized_dataset["train"]["text"]

    test_set = tokenized_dataset["test"].train_test_split(test_size=0.001, shuffle=True)

    test = test_set["train"]["text"]
    validation = test_set["test"]["text"]

    # hyperparameters

    model_config = TransformerConfig(
        n_vocab=encoder.n_vocab + 1, # NOTE: add 1 for <EOS> token
        d_model=1024,
        d_query=64,
        n_heads=16,
        n_layers=16,
        d_up=512,
        device=device
    )

    n_epochs    = 10
    batch_size  = 8
    accum_steps = 4

    def collate_fn_padding(batch):
        batch = pad_sequence(batch, batch_first=True)
        return batch

    checkpoint_path = "checkpoint.pt"

    model = Transformer(model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

    sampler = CheckpointRandomSampler(train, batch_size, checkpoint)

    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_padding)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padding)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padding)

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