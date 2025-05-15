import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from rl.stackRNN import StackAugmentedRNN
from rl.data import GeneratorData
from rl.utils import canonical_smiles
import matplotlib.pyplot as plt
import seaborn as sns
from rl.data import PredictorData
from rl.utils import get_desc, get_fp
from rl.predictor import QSAR
from rl.predictor import process_smiles_to_morgan_tensor as get_features
from mordred import Calculator, descriptors
import torch
import torch.nn.functional as F


class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward):

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def policy_gradient(self, data, n_batch=10, gamma=0.97, std_smiles=False, grad_clipping=None, prime_str=None, **kwargs):
        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        
        for _ in range(n_batch):
            reward = 0
            trajectory = '<>'
            while reward == 0:
                trajectory = self.generator.evaluate(data, prime_str=prime_str)
                if std_smiles:
                    try:
                        mol = Chem.MolFromSmiles(trajectory[1:-1])
                        trajectory = '<' + Chem.MolToSmiles(mol) + '>'
                        reward = self.get_reward(trajectory[1:-1], 
                                                 self.predictor, 
                                                 **kwargs)
                    except:
                        reward = 0
                else:
                    reward = self.get_reward(trajectory[1:-1],
                                             self.predictor, 
                                             **kwargs)

            trajectory_input = data.char_tensor(trajectory)
            discounted_reward = reward
            total_reward += reward


            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack()
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(trajectory_input[p], 
                                                       hidden, 
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()


def get_result(my_string):

    calc = Calculator(descriptors, ignore_3D=True)

    def predictions_to_numpy(predictions):
        import torch
        import numpy as np

        if isinstance(predictions, torch.Tensor):
            return predictions.detach().cpu().numpy()

        elif isinstance(predictions, list):
            processed = []
            for p in predictions:
                if isinstance(p, torch.Tensor):
                    p_np = p.detach().cpu().numpy()
                    processed.append(p_np.item() if p_np.size == 1 else p_np)
                else:
                    processed.append(p)
            return np.array(processed, dtype=np.float32)

        else:
            return np.array(predictions, dtype=np.float32)

    def plot_hist(prediction, n_to_generate):
        import torch

        # Convert predictions to a numpy array of floats
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        elif isinstance(prediction, list):
            if all(isinstance(p, torch.Tensor) for p in prediction):
                prediction = np.array([p.item() for p in prediction], dtype=np.float32)
            else:
                prediction = np.array(prediction, dtype=np.float32)
        else:
            prediction = np.array(prediction, dtype=np.float32)

        # Flatten to 1D
        prediction_flat = prediction.reshape(-1)

        # Stats
        print("Mean value of predictions:", prediction_flat.mean())
        print("Proportion of valid SMILES:", len(prediction_flat) / n_to_generate)

        # Plot
        # ax = sns.kdeplot(prediction_flat, shade=True)
        # ax.set(xlabel='Predicted pIC50', 
        #        title='Distribution of predicted pIC50 for generated molecules')
        # plt.show()

    def estimate_and_update(generator, predictor, n_to_generate, **kwargs):
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

        sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
        unique_smiles = list(np.unique(sanitized))[1:]
        smiles, prediction, nan_smiles = predictor.predict(unique_smiles)  

        plot_hist(predictions_to_numpy(prediction), n_to_generate)

        return smiles, prediction

    model_path = 'C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/checkpoint_biggest_rnn'

    gen_data_path = 'C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/123.smi'
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
            '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
            '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                            cols_to_read=[0], keep_header=True, tokens=tokens)

    # print(f"\n\n\n\n\n\n-------------------------{gen_data}----------------------\n\n\n\n\n\n\n")


    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                    output_size=gen_data.n_characters, layer_type=layer_type,
                                    n_layers=1, is_bidirectional=False, has_stack=True,
                                    stack_width=stack_width, stack_depth=stack_depth, 
                                    use_cuda=None, 
                                    optimizer_instance=optimizer_instance, lr=lr)

    my_generator.load_model(model_path)




    n_hidden = 512
    batch_size = 128
    num_epochs = 50
    lr = 0.005
    def identity(input):
        return input

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
            'activation': [F.relu, identity],
            'dropout': 0.0
        }
    }

    predictor=QSAR(model_params)

    predictor.load_model('C:/Users/nani2/Downloads/project-bolt-sb1-vxvushvx/project/rl/qsar_model_pic50.pt')

    smiles_unbiased, prediction_unbiased = estimate_and_update(my_generator, predictor, n_to_generate=10)


    my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, 
                                        hidden_size=hidden_size,
                                        output_size=gen_data.n_characters, 
                                        layer_type=layer_type,
                                        n_layers=1, is_bidirectional=False, has_stack=True,
                                        stack_width=stack_width, stack_depth=stack_depth, 
                                        use_cuda=None, 
                                        optimizer_instance=optimizer_instance, lr=lr)

    my_generator_max.load_model(model_path)



    # Setting up some parameters for the experiment
    n_to_generate = 5
    n_policy_replay = 5
    n_policy = 5
    n_iterations = 1

    def simple_moving_average(previous_values, new_value, ma_window_size=10):
        # Convert previous values to floats
        prev_vals = [v.item() if isinstance(v, torch.Tensor) else v for v in previous_values[-(ma_window_size-1):]]
        new_val = new_value.item() if isinstance(new_value, torch.Tensor) else new_value

        value_ma = np.sum(prev_vals) + new_val
        value_ma = value_ma / (len(prev_vals) + 1)
        return value_ma


    def get_reward_max(smiles, predictor, invalid_reward=0.0, get_features=get_fp):
        mol, prop, nan_smiles = predictor.predict([smiles])
        if len(nan_smiles) == 1:
            return invalid_reward
        return np.exp(prop[0] / 3)

    RL_max = Reinforcement(my_generator_max, predictor, get_reward_max)

    rewards_max = []

    rl_losses_max = []

    smile_result=[]
    pic50_result=[]
    my_data=my_string
    for i in range(n_iterations):
        for j in trange(n_policy, desc='Policy gradient...'):
            reference_smiles = "<"+my_data
            cur_reward, cur_loss = RL_max.policy_gradient(gen_data, get_features=get_fp, prime_str=reference_smiles)

            # Apply moving average
            rewards_max.append(simple_moving_average(rewards_max, cur_reward))
            rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))


        # Sample molecules and print
        smiles_cur, prediction_cur = estimate_and_update(RL_max.generator,predictor,10,get_features=get_fp)
        print('Sample trajectories:')
        # smile_result=[]
        # pic50_result=[]
        count=0
        for sm in smiles_cur:
            smile_result.append([sm,prediction_cur[0][count].item()])
            # pic50_result.append(prediction_cur[0][count].item())
            # print(sm," --- ",prediction_cur[0][count].item())
            count+=1
    # Save the RL-trained generator
    RL_max.generator.save_model("./rl_trained_generator.pt")
    return smile_result
            
            
if __name__=="__main__":
    s=get_result("CC(=O)Oc1ccccc1C(=O)O")
    print(s)