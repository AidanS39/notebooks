import torch
import time

from model import TransformerCheckpoint
from model import Transformer
from model import generate_response

def main():
    # torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device, flush=True)
    
    checkpoint = TransformerCheckpoint.load("./models/model_2048_8_8_2048.pt")

    encoder = checkpoint.encoder
    config = checkpoint.config
    model_state = checkpoint.model_state

    model = Transformer(config).to(device)
    model.load_state_dict(model_state)
    model = torch.compile(model)

    prompt = "Once upon a time, there was a bear in the forest."

    print("generating sequence...")
    start_time = time.time()
    response = generate_response(model, encoder=encoder, device=device, prompt=prompt, temp=1.5, k=3)
    print(f"\ngenerated sequence in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()