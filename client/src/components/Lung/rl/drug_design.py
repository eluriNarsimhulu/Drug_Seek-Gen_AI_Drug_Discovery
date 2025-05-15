from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import IPythonConsole

from itertools import combinations

import IPython
from IPython.display import display, Image
from PIL import Image

import numpy as np
import pandas as pd

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import re
import random

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import MolToImage

from PIL import Image

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("S:/my_works/reactjs/practice/practice2/src/rl/SMILES_Big_Data_Set.csv")

class DRLAgent:
    def __init__(self, state_size, action_size, selected_X_train):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.selected_X_train = selected_X_train
        self.model = self._build_model(selected_X_train)
        self.target_model = self._build_model(selected_X_train) 
        self.update_target_model()

    def _build_model(self, selected_X_train):
        model = Sequential()
        model.add(Dense(32, input_shape=(selected_X_train.shape[1],), activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='linear')) 
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def update_target_model(self):
        """Update the target model weights with the current model weights"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay memory

        Args:
            state (ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (ndarray): Next state
            done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action given the current state

        Args:
            state (ndarray): Current state

        Returns:
            action (int): Chosen action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        if state is None:
            state = np.zeros((1, self.state_size))
        else:
            state = state.reshape(1, self.state_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Train the model by replaying experiences from the replay memory

        Args:
            batch_size (int): Size of the minibatch
        """
        # Sample a minibatch of experiences from the replay memory
        minibatch = random.sample(self.memory, batch_size)
        # print(minibatch)
        for state, action, reward, next_state, done in minibatch:
            if state is not None:
                if not done:
                    # Calculate the target value using the target model
                    target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
                else:
                    target = reward
                # Make the agent approximately map the current state to future discounted reward
                target_f = self.model.predict(state)
                target_f[0][action] = target
                # Train the model using the current state and target value
                self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load the model weights from a file"""
        self.model.load_weights(name)

    def save(self, name):
        """Save the model weights to a file"""
        self.model.save_weights(name)
        


def preprocess_smiles(smiles):
    """
    Preprocess the SMILES string by removing salts and stereochemistry information

    Args:
        smiles (str): SMILES string

    Returns:
        preprocessed_smiles (str): Preprocessed SMILES string
    """
    # Remove salts
    preprocessed_smiles = re.sub(r'\[.*?\]', '', smiles)
    # Remove stereochemistry information
    preprocessed_smiles = re.sub(r'[@]\S*', '', preprocessed_smiles)
    # print(preprocessed_smiles)
    return preprocessed_smiles


def calculate_molecular_properties(smiles):
    """
    Calculate molecular properties of a compound given its SMILES string

    Args:
        smiles (str): SMILES string

    Returns:
        properties (dict): Dictionary of molecular properties
    """
    molecule = Chem.MolFromSmiles(smiles)
    properties = {}

    if molecule is not None:
        properties['Molecular Weight'] = Descriptors.MolWt(molecule)
        properties['LogP'] = Descriptors.MolLogP(molecule)
        properties['H-Bond Donor Count'] = Descriptors.NumHDonors(molecule)
        properties['H-Bond Acceptor Count'] = Descriptors.NumHAcceptors(molecule)

    return properties

def get_closest_smiles(predicted_features, X_train, smiles_list):
    """
    Find the closest SMILES string based on predicted molecular features and training data features.

    Args:
        predicted_features (ndarray): The features predicted by the model.
        X_train (ndarray): The training feature matrix (properties of the molecules in training set).
        smiles_list (list): List of SMILES strings corresponding to X_train.

    Returns:
        str: Closest SMILES string based on similarity to predicted features.
    """
    # Ensure X_train has the same number of features as predicted_features
    if X_train.shape[1] != predicted_features.shape[1]:
        # Here, you can slice X_train to match predicted_features' shape if needed
        X_train = X_train[:, :predicted_features.shape[1]]
        # print(X_train)

    # Calculate similarity (e.g., cosine similarity between the predicted features and training set features)
    similarities = cosine_similarity(predicted_features, X_train)
    # similarities=similarities*100
    # similarities=int(similarities[0][0])
    # print(similarities)
    
    # Find the index of the most similar molecule
    closest_index = np.argmax(similarities)
    # closest_index = similarities
    # print(closest_index)
    return smiles_list[closest_index]




def main(value):
    smiles = value
    preprocessed_smiles = preprocess_smiles(smiles)
    print(f"Preprocessed SMILES: {preprocessed_smiles}")
    
    properties = calculate_molecular_properties(preprocessed_smiles)
    print(f"Molecular Properties: {properties}")
    
    # Convert properties to a NumPy array and reshape for the model
    selected_X_train = np.array(list(properties.values())).reshape(1, -1)
    
    # Define a dummy list of SMILES (just an example, you'd have your actual dataset here)
    smiles_list = [smile for smile in df["SMILES"]]  # Add more SMILES here as needed
    # print(smiles_list)
    # for s in smiles_list:
    #     print(s)
    X_train = np.array([list(calculate_molecular_properties(smiles).values()) for smiles in smiles_list])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Add more property vectors here as needed
    # print(X_train)
    
    # Initialize the agent with the feature dimensions
    
    # Get model's output for the selected input (predicted features)
    
    # Get the closest SMILES string based on predicted features
    similar_smiles_string=[]
    for i in range(5):
        agent = DRLAgent(state_size=selected_X_train.shape[1], action_size=3, selected_X_train=selected_X_train)
        predicted_features = agent.model.predict(selected_X_train)
        print(f"Predicted Features: {predicted_features}")
        
        predicted_features_scaled = scaler.transform(predicted_features)
        smile_string_my_=get_closest_smiles(predicted_features_scaled, X_train_scaled, smiles_list)
        similar_smiles_string.append(smile_string_my_)

        # agent.model.save("hello.h5")
        # for item in similar_smiles_string:
            # print(f"Closest SMILES: ",preprocess_smiles(item))
            # print("\n\n\n")
        # print(smile_string_my_)
        img1=Chem.MolFromSmiles(smile_string_my_)
        img = Draw.MolsToImage([img1], molsPerRow=1, subImgSize=(400,400),
                             legends=[f'{smile_string_my_}'], 
                             returnPNG=False).save(f"S:/my_works/reactjs/practice/practice2/src/images/{i}.png")
        
    return similar_smiles_string

if __name__ == '__main__':
    # for item in df["SMILES"][0]:
        # print(random.randint(0,len(df['SMILES'])))
        main(df["SMILES"][0])
        # main(df['SMILES'][random.randint(0,len(df['SMILES']))])
