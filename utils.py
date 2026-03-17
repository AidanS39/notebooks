import time
import argparse
import torch
import warnings
from pathlib import Path

class EpochLog():
    def __init__(self, epoch):
        self.epoch = epoch
        self.batch_logs = list()
    def add_batch_log(self, batch, loss):
        self.batch_logs.append((batch, loss))

class TrainLog():
    def __init__(self):
        self.epoch_logs: list[EpochLog] = list()
        self.start_time = None
        self.end_time = None
    def add_epoch_log(self, epoch_log: EpochLog):
        self.epoch_logs.append(epoch_log)
    def start_timer(self):
        if self.start_time == None:
            self.start_time = time.time()
    def stop_timer(self):
        if self.end_time == None:
            self.end_time = time.time()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda") 
    else:
        warnings.warn("CUDA device not available, will use CPU instead.", UserWarning)
        return torch.device("cpu")
    
def slow_print(sequence: str, char_delay: float = 0.02, word_delay: float = 0.1, delay: bool = True):

    if delay == True:
        sequence = sequence.split(sep=" ")
        for i, word in enumerate(sequence):
            for letter in word:
                print(f"{letter}", end="", flush=True)
                time.sleep(char_delay)
            if i < len(sequence) - 1:
                print(" ", end="", flush=True)
            time.sleep(word_delay)
    else:
        print(sequence, end="", flush=True)

def get_model_files(models_path: str):
    model_dir = Path(models_path)
    model_files = sorted([f.name for f in model_dir.glob("*.pt")])
    return model_files