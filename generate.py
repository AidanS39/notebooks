import torch
import time
import argparse
import warnings

from model import TransformerCheckpoint
from model import Transformer
from model import generate_response

from utils import get_device
from utils import slow_print
from utils import get_model_files

def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--help", action="help", help="help menu")

    parser.add_argument("-f", "--models_path", type=str, default="./models", help="path to models folder")
    parser.add_argument("--no-delay", action="store_false", dest="delay", help="disable print delay")

    args = parser.parse_args()

    return args

def main():
    # torch.manual_seed(42)
    warnings.filterwarnings('ignore', category=UserWarning)
    torch.set_float32_matmul_precision('high')

    args = get_arguments()

    model_files = get_model_files(args.models_path)

    print("Available models:")
    for i, key in enumerate(model_files):
        print(f"{i + 1}  - {key}")
    model_index = int(input("Choose a model: "))

    model_path = f"{args.models_path}/{model_files[model_index - 1]}"

    print(f"Loading {model_files[model_index - 1]}...")
    
    device = get_device()

    checkpoint = TransformerCheckpoint.load(model_path)

    slow_print(
        r"""
 ________________________________________________________________________________
|       __________  ___    _   _______ __________  ____  __  _____________       |
|      /_  __/ __ \/   |  / | / / ___// ____/ __ \/ __ \/  |/  / ____/ __ \      |
|       / / / /_/ / /| | /  |/ /\__ \/ /_  / / / / /_/ / /|_/ / __/ / /_/ /      |
|      / / / _, _/ ___ |/ /|  /___/ / __/ / /_/ / _, _/ /  / / /___/ _, _/       |
|     /_/ /_/ |_/_/  |_/_/ |_//____/_/    \____/_/ |_/_/  /_/_____/_/ |_|        |
|________________________________________________________________________________|
""", char_delay=0.005, word_delay=0, delay=args.delay)
    
    slow_print("\nWelcome to the TRANSFORMER Program!\n", delay=args.delay)

    slow_print("""
Note: The first response will be slower than subsequent ones, as the model fully loads 
      during the initial forward pass. Later responses will be significantly faster.

To generate a response: type out a prompt, then press ENTER.
To exit the program: To exit, press `ENTER` with an empty prompt or type `exit`.
""", delay=args.delay)

    encoder = checkpoint.encoder
    config = checkpoint.config
    model_state = checkpoint.model_state

    model = Transformer(config).to(device)
    model.load_state_dict(model_state)
    model = torch.compile(model)

    keep_prompting = True

    while keep_prompting:
        prompt = input("Write the beginning of a story: ")

        start_time = time.time()
        response = generate_response(model, encoder=encoder, device=device, prompt=prompt, temp=1.5, k=3)

        if response == None:
            slow_print("\nGoodbye!\n", delay=args.delay)
            keep_prompting = False
        else:
            print(f"\n\ngenerated sequence in {time.time() - start_time} seconds\n")

if __name__ == "__main__":
    main()