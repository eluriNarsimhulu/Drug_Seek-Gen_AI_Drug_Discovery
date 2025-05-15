import torch
from rl.data import GeneratorData
from rl.stackRNN import StackAugmentedRNN
from rl.predictor import QSAR
from rl.utils import canonical_smiles
import numpy as np
from tqdm import tqdm
# from mordred import Calculator, descriptors

def load_models_and_data():
    # Load generator data setup
    gen_data_path = 'C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/123.smi'
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                             cols_to_read=[0], keep_header=True, tokens=tokens)

    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    # Load trained RL generator
    generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                  hidden_size=hidden_size,
                                  output_size=gen_data.n_characters,
                                  layer_type=layer_type,
                                  n_layers=1, is_bidirectional=False, has_stack=True,
                                  stack_width=stack_width, stack_depth=stack_depth,
                                  use_cuda=None,
                                  optimizer_instance=optimizer_instance, lr=lr)

    generator.load_model("C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/rl_trained_generator.pt")

    # Load predictor
    n_hidden = 512
    def identity(input): return input

    model_params = {
        'embedding': "finger_prints",
        'embedding_params': {'embedding_dim': n_hidden, 'fingerprint_dim': 2048},
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
            'activation': [torch.nn.functional.relu, identity],
            'dropout': 0.0
        }
    }

    predictor = QSAR(model_params)
    predictor.load_model('C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/qsar_model_pic50.pt')

    return gen_data, generator, predictor


def generate_similar_molecules(input_smile, n_to_generate, gen_data, generator, predictor):
    # print(f"\nGenerating molecules similar to: {input_smile}\n")
    prime_str = "<" + input_smile

    generated = set()
    max_attempts = n_to_generate * 10  # Avoid infinite loops in edge cases
    attempts = 0

    pbar = tqdm(total=n_to_generate, desc="Generating unique similar molecules...")

    while len(generated) < n_to_generate and attempts < max_attempts:
        smiles = generator.evaluate(gen_data, predict_len=120, prime_str=prime_str)[1:-1]
        sanitized = canonical_smiles([smiles], sanitize=False, throw_warning=False)[0]
        if sanitized and sanitized not in generated:
            generated.add(sanitized)
            pbar.update(1)
        attempts += 1

    pbar.close()

    if len(generated) < n_to_generate:
        print(f"\n⚠️ Only generated {len(generated)} unique molecules after {max_attempts} attempts.\n")

    unique_smiles = list(generated)

    smiles, prediction, _ = predictor.predict(unique_smiles)

    # Combine smiles and predictions and sort by pIC50 descending
    results = list(zip(smiles, prediction[0]))
    results.sort(key=lambda x: x[1].item(), reverse=True)

    # print(f"\nTop {len(results)} generated molecules (ranked by predicted pIC50):\n")
    
    my_result=[]
    for sm, p in results:
        my_result.append([sm,p.item()])
        # print(f"{sm} --- pIC50: {p.item()}")
    return my_result


def my_pic50(smile):
    gen_data, generator, predictor = load_models_and_data()

    # 🚀 Only generate molecules similar to a given SMILES
    input_smile = smile
    output=generate_similar_molecules(input_smile, 5, gen_data, generator, predictor)
    return output

if __name__ == "__main__":
    gen_data, generator, predictor = load_models_and_data()

    # 🚀 Only generate molecules similar to a given SMILES
    input_smile = "CC(=O)Oc1ccccc1C(=O)O"
    output=generate_similar_molecules(input_smile, 5, gen_data, generator, predictor)
    # print(f"\nTop {len(x)} generated molecules (ranked by predicted pIC50):\n") 
    print(output)


