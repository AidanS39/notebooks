import torch
import time

from model import TransformerCheckpoint
from model import Transformer
from model import generate_response

def main():
    # torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    checkpoint = TransformerCheckpoint.load("checkpoint.pt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    encoder = checkpoint.encoder
    config = checkpoint.config
    model_state = checkpoint.model_state

    model = Transformer(config).to(device)
    model.load_state_dict(model_state)
    model = torch.compile(model)

    prompt = "One day, there was a bear looking for honey."

    print("generating sequence...")
    start_time = time.time()
    response = generate_response(model, encoder=encoder, device=device, prompt=prompt, temp=1.5, k=2)
    print(f"\ngenerated sequence in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()