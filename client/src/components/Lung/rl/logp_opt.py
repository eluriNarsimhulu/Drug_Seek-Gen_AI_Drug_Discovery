import pickle
import torch
import torch.nn as nn
import numpy as np
import os
from rdkit import Chem
import torch.nn.functional as F
from rl.rnn_predictor import RNNPredictor
from rl.stackRNN import StackAugmentedRNN
from rl.data import GeneratorData
from rl.utils import canonical_smiles
from rl.main_rl import Reinforcement

# Check CUDA availability
use_cuda = torch.cuda.is_available()

# Data paths
gen_data_path = 'C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/123.smi'
model_path = 'C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/checkpoint_biggest_rnn'
checkpoint_path = "C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/fold_1.pkl"

# SMILES tokens
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

# Data loader
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)

# Generator setup
hidden_size, stack_width, stack_depth = 1500, 1500, 200
generator = StackAugmentedRNN(
    input_size=gen_data.n_characters,
    hidden_size=hidden_size,
    output_size=gen_data.n_characters,
    layer_type='GRU',
    n_layers=1,
    is_bidirectional=False,
    has_stack=True,
    stack_width=stack_width,
    stack_depth=stack_depth,
    use_cuda=use_cuda,
    optimizer_instance=torch.optim.Adadelta,
    lr=0.001
)
generator.load_model(model_path)

# Predictor config
n_hidden = 512
predictor_tokens = tokens + [' ']
model_params = {
    'embedding': "finger_prints",
    'embedding_params': {
        'embedding_dim': n_hidden,
        'fingerprint_dim': 2048
    },
    'encoder': "RNNEncoder",
    'encoder_params': {
        'input_size': 2048,
        'layer': "GRU",
        'encoder_dim': n_hidden,
        'n_layers': 2,
        'dropout': 0.8
    },
    'mlp': "mlp",
    'mlp_params': {
        'input_size': n_hidden,
        'n_layers': 2,
        'hidden_size': [n_hidden, 1],
        'activation': [F.relu, lambda x: x],
        'dropout': 0.0
    }
}
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

predictor = RNNPredictor(model_params, checkpoint_path, predictor_tokens)

# Reward function (not changed)
def get_reward_logp(smiles, predictor, invalid_reward=0.0):
    mol, prop, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    return 11.0 if 1.0 <= prop[0] <= 4.0 else 1.0

# Reinforcement setup
rl_agent = Reinforcement(generator, predictor, get_reward_logp)

# Faster generation + prediction
def generate_similar_smiles(input_smiles, n_to_generate=50):
    generated = [generator.evaluate(gen_data, prime_str=input_smiles, predict_len=120)[1:-1]
                 for _ in range(n_to_generate)]

    # Faster deduplication before prediction
    unique_smiles = list(set(generated))

    # Skip sanitization if unnecessary for your model
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, use_tqdm=False)

    # Filter in one pass
    valid_pairs = [(sm, logp) for sm, logp in zip(smiles, prediction) if 1.0 <= logp <= 4.0]

    return valid_pairs

# Main run function, optimized
def my_logp(smile):
    input_smiles = smile

    # Generate once and pick top 5
    valid_pairs = generate_similar_smiles(input_smiles, n_to_generate=50)

    my_result=[]
    # Sort descending by LogP and pick top 5
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    valid_pairs=valid_pairs[:5]
    for sm, logp in valid_pairs:
        my_result.append([sm,logp])
    return my_result

# Example run
if __name__ == "__main__":
    data = my_logp("CC(=O)Oc1ccccc1C(=O)O")
    print(data)
    # for sm, logp in data:
    #     print(f"{sm}  |  LogP: {logp:.2f}")
