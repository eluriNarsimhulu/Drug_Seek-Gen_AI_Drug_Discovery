import io
import base64
import hashlib
import numpy as np
import cv2
import tensorflow as tf
from patchify import patchify
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from metrics import dice_loss, dice_coef
from PIL import Image
import torch
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rl.final import my_pic50
from rl.logp_opt import my_logp
import joblib


# from flask import Flask, request, jsonify
import requests
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs
# from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Apply CORS for all routes

# =========================
# IMAGE SEGMENTATION MODEL
# =========================

# Load the trained segmentation model
seg_model = None
image_cache = {}  # In-memory cache: {hash: {original, prediction, overlay}}
# print(image_cache)

cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16,
    "num_patches": (256**2) // (16**2),
    "flat_patches_shape": (
        (256**2) // (16**2),
        16 * 16 * 3
    )
}

def load_segmentation_model():
    global seg_model
    model_path = "files/models.keras"
    seg_model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
    print("Segmentation model loaded successfully!")

def get_file_hash(file_stream):
    """Generate SHA256 hash from in-memory stream"""
    hasher = hashlib.sha256()
    buf = file_stream.read()
    hasher.update(buf)
    file_stream.seek(0)
    return hasher.hexdigest(), buf

def preprocess_image(image_bytes):
    """Convert bytes to model input"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((cf["image_size"], cf["image_size"]))
    orig_image = np.array(image)
    x = orig_image / 255.0

    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    return patches, orig_image

def predict_mask(patches):
    pred = seg_model.predict(patches, verbose=0)[0]
    return pred

def encode_image(image_array):
    """Encode numpy image to base64"""
    _, buffer = cv2.imencode('.png', image_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"

# =========================
# SMILES PREDICTION MODEL
# =========================

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open("updated_vocab.json", "r") as f:
    vocab = json.load(f)

# Load tokenizer from vocab and merges
bpe_model = BPE.from_file("updated_vocab.json", "merges.txt")
tokenizer = Tokenizer(bpe_model)
tokenizer.add_special_tokens(['<mask>'])

# Define SMILES Model Architecture
class RoBERTaEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        x = self.token_embed(input_ids) + self.position_embed(positions)
        return self.dropout(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, D = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return self.dropout(self.out_proj(out))

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden=512):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class RoBERTaForMaskedLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, max_len=128):
        super().__init__()
        self.embedding = RoBERTaEmbedding(vocab_size, embed_dim, max_len)
        self.encoder = nn.Sequential(*[
            CustomTransformerBlock(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)

# Load SMILES model
smiles_model = RoBERTaForMaskedLM(len(vocab))
smiles_model.load_state_dict(torch.load("model.pth", map_location=device))
smiles_model.to(device)
smiles_model.eval()

class SMILESDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_len=128):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.smiles[idx])
        input_ids = encoded.ids
        input_ids += [vocab["[PAD]"]] * (self.max_len - len(input_ids))
        return torch.tensor(input_ids[:self.max_len], dtype=torch.long)

def encode_input(smiles, tokenizer, vocab):
    tokens = []
    print(f"Original SMILES: {smiles}")  # Print the original SMILES
    encoded = tokenizer.encode(smiles)
    print(f"Encoded Tokens: {encoded.tokens}")  # Print the tokenized representation
    print(f"Encoded Token IDs: {encoded.ids}")  # Print the corresponding token IDs

    for token in encoded.tokens:
        if token == "<mask>":
            tokens.append(vocab["<mask>"])
        else:
            tokens.append(vocab.get(token, vocab["[UNK]"]))
    return torch.tensor(tokens).unsqueeze(0).to(device)

def predict_smiles(smiles_input):
    input_ids = encode_input(smiles_input, tokenizer, vocab)
    print(f"Input IDs: {input_ids}")  # Print the input IDs tensor

    mask_token_id = vocab["<mask>"]
    print(f"Mask Token ID: {mask_token_id}")  # Print the ID for <mask>

    mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    print(f"Mask Indices: {mask_indices}")  # Print the indices where mask is found

    if len(mask_indices) == 0:
        return "No <mask> token found in input!"

    mask_index = mask_indices[0].item()

    with torch.no_grad():
        logits = smiles_model(input_ids)
        mask_logits = logits[0, mask_index]

        # Convert to probabilities
        probs = F.softmax(mask_logits, dim=-1)

        # Get top 5 predictions
        topk_probs, topk_indices = torch.topk(probs, k=5)
        topk_probs = topk_probs.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        results = []
        for prob, token_id in zip(topk_probs, topk_indices):
            token = list(vocab.keys())[list(vocab.values()).index(token_id)]
            results.append(f"{token} ({prob:.4f})")

        # Combine top 5 into a single string for display
        return "Top 5 Predictions:\n" + "\n".join(results)

# =========================
# MOLECULAR PROPERTY PREDICTION
# =========================

def smiles_to_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image for {smiles}: {str(e)}")
        return None

# ====================================
# ViT Model Definition (from second app.py)
# ====================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embedding_dim=48):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=attn_dropout, 
            batch_first=True
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm, need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=48, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embedding_dim=48, num_transformer_layers=12, num_heads=4,
                 mlp_size=3072, attn_dropout=0, mlp_dropout=0.1, embedding_dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout, attn_dropout)
            for _ in range(num_transformer_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position_embedding)
        for block in self.encoder_blocks:
            x = block(x)
        return self.classifier(x[:, 0])

# Load ViT model
vit = None
le = None

def load_vit_model():
    global vit, le
    # Load model config from pkl file
    model_data = joblib.load("vit_model.pkl")
    
    # Initialize model
    vit = ViT(**model_data["model_config"])
    
    # Load state dict from pth file
    vit.load_state_dict(torch.load(model_data["model_state_dict"], map_location=torch.device('cpu')))
    vit.eval()
    
    le = model_data["label_encoder"]
    print("ViT model loaded successfully!")

def morgan_to_image(x):
    flat = np.pad(x, (0, 3 * 32 * 32 - len(x)), constant_values=0)
    return flat.reshape(3, 32, 32)

# =========================
# FLASK ROUTES
# =========================


def download_pdb_and_ligand(ECnumber, LIGAND_ID):
    q1 = attrs.rcsb_polymer_entity.rcsb_ec_lineage.id == ECnumber
    q2 = TextQuery(LIGAND_ID)

    query = q1 & q2
    results = list(query())

    if not results:
        return None, None

    pdb_id = results[0].lower()
    ligand_id = LIGAND_ID.lower()

    pdb_response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    ligand_response = requests.get(f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf")

    if pdb_response.status_code != 200 or ligand_response.status_code != 200:
        return None, None

    pdb_content = pdb_response.text
    ligand_content = ligand_response.text

    return pdb_content, ligand_content

@app.route('/auto/process', methods=['POST'])
def process_ligand():
    try:
        data = request.get_json()
        ECnumber = data.get('ECnumber')
        LIGAND_ID = data.get('LIGAND_ID')

        if not ECnumber or not LIGAND_ID:
            return jsonify({"error": "ECnumber and LIGAND_ID are required"}), 400

        pdb_content, ligand_content = download_pdb_and_ligand(ECnumber, LIGAND_ID)

        if not pdb_content or not ligand_content:
            return jsonify({"error": "Failed to download PDB or ligand data"}), 400

        return jsonify({
            "pdb_content": pdb_content
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/upload/lung', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_hash, file_bytes = get_file_hash(file.stream)

    # Check if the image is already in cache
    if file_hash in image_cache:
        print("Result is from Cache")
        print ("Hash" , file_hash)
        return jsonify({
            **image_cache[file_hash], 
            "from_cache": True
        }), 200

    try:
        patches, orig_image = preprocess_image(file_bytes)
        pred_mask = predict_mask(patches)

        pred_rgb = np.concatenate([pred_mask] * 3, axis=-1) * 255
        pred_rgb = pred_rgb.astype(np.uint8)

        red_mask = np.zeros_like(orig_image)
        red_mask[:, :, 2] = pred_rgb[:, :, 0]
        overlay_result = cv2.addWeighted(orig_image, 1, red_mask, 0.5, 0)

        # Convert images to base64
        result_data = {
            "original": encode_image(orig_image),
            "prediction": encode_image(pred_rgb),
            "overlay": encode_image(overlay_result),
            "from_cache": False  # Newly processed result
        }
        # Cache result
        image_cache[file_hash] = result_data
        # print(image_cac
        # he)
        print("Cached result for image hash:", file_hash)
        print("Original (base64):", result_data["original"][:100], "...")
        print("Prediction (base64):", result_data["prediction"][:100], "...")
        print("Overlay (base64):", result_data["overlay"][:100], "...")
        
        return jsonify(result_data), 200

    except Exception as e:
        print("Processing Error:", e)
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/chemberta/predict', methods=['POST'])
def predict_view():
    data = request.get_json()
    smiles_input = data.get('user_input', '')
    if not smiles_input:
        return jsonify({'error': 'No input provided'}), 400

    predicted_smiles = predict_smiles(smiles_input)

    return jsonify({
        "prediction": predicted_smiles
    })

@app.route('/api/reinforce/predict', methods=['POST'])
def molecular_predict():
    try:
        data = request.get_json()
        
        if not data or 'smiles' not in data or 'type' not in data:
            return jsonify({
                'error': 'No SMILES data or prediction type provided'
            }), 400
            
        smiles = data['smiles']
        pred_type = data['type']
        
        # Validate SMILES input (basic validation)
        if not smiles or len(smiles) < 2:
            return jsonify({
                'error': 'Invalid SMILES input'
            }), 400
        
        # Call the appropriate prediction function
        if pred_type == 'pic50':
            results = my_pic50(smiles)
            formatted_results = [{
                'smiles': item[0],
                'pic50': round(item[1], 3),
                'image': smiles_to_image(item[0])
            } for item in results]
        elif pred_type == 'logp':
            results = my_logp(smiles)
            formatted_results = [{
                'smiles': item[0],
                'logp': round(item[1], 3),
                'image': smiles_to_image(item[0])
            } for item in results]
        else:
            return jsonify({
                'error': 'Invalid prediction type'
            }), 400
        
        # Add image for the input SMILES
        input_image = smiles_to_image(smiles)
        
        return jsonify({
            'input_smiles': smiles,
            'input_image': input_image,
            'results': formatted_results,
            'type': pred_type
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

# Add the ViT model prediction route from second app.py
@app.route('/vit/predict', methods=['POST'])
def vit_predict():
    try:
        data = request.get_json()
        smiles = data.get('smiles', '')
        
        # Convert SMILES to fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES string"}), 400
            
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.array(fingerprint, dtype=np.float32)
        fp_array = fp_array / (fp_array.max() + 1e-6)
        
        # Convert to image and predict
        image = morgan_to_image(fp_array)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = vit(image_tensor)
            predicted_class_idx = output.argmax(1).item()
            predicted_label = le.inverse_transform([predicted_class_idx])[0]
            
        return jsonify({
            "smiles": smiles,
            "predicted_class": predicted_label
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Combined ML Services</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                button { background-color: #4CAF50; color: white; padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #45a049; }
                input, select { padding: 8px; margin-right: 10px; border: 1px solid #ddd; border-radius: 4px; }
                .result-container { margin-top: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 4px; }
                .molecules { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }
                .molecule-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; width: 200px; }
                .molecule-img { width: 100%; max-height: 150px; object-fit: contain; }
            </style>
        </head>
        <body>
            <h1>Multi-Modal ML Services</h1>
            
            <div class="section">
                <h2>Image Segmentation Service</h2>
                <p>Upload an image for segmentation</p>
                <form action="/upload/lung" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <input type="submit" value="Upload">
                </form>
            </div>
            
            <div class="section">
                <h2>SMILES Prediction Service</h2>
                <p>Enter SMILES with &lt;mask&gt; token</p>
                <form id="smilesForm">
                    <input type="text" id="smilesInput" placeholder="Enter SMILES with <mask>">
                    <button type="submit">Predict</button>
                </form>
                <div id="smiles-result" class="result-container"></div>
            </div>
            
            <div class="section">
                <h2>Molecular Property Prediction</h2>
                <p>Predict pIC50 or LogP for molecules</p>
                <form id="molPropForm">
                    <input type="text" id="molInput" placeholder="Enter SMILES string" required>
                    <select id="propType">
                        <option value="pic50">pIC50</option>
                        <option value="logp">LogP</option>
                    </select>
                    <button type="submit">Predict</button>
                </form>
                <div id="mol-result" class="result-container">
                    <div id="input-molecule"></div>
                    <div id="result-molecules" class="molecules"></div>
                </div>
            </div>
            
            <div class="section">
                <h2>ViT Model Classification</h2>
                <p>Classify molecules using Vision Transformer</p>
                <form id="vitForm">
                    <input type="text" id="vitInput" placeholder="Enter SMILES string" required>
                    <button type="submit">Classify</button>
                </form>
                <div id="vit-result" class="result-container"></div>
            </div>
            
            <script>
                // SMILES Prediction
                document.getElementById('smilesForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const userInput = document.getElementById('smilesInput').value;
                    const resultElement = document.getElementById('smiles-result');
                    resultElement.innerText = "Processing...";
                    
                    try {
                        const response = await fetch('/chemberta/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ user_input: userInput }),
                        });
                        
                        const data = await response.json();
                        resultElement.innerText = data.prediction;
                    } catch (error) {
                        console.error('Error:', error);
                        resultElement.innerText = 'Error processing request';
                    }
                });
                
                // Molecular Property Prediction
                document.getElementById('molPropForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const smiles = document.getElementById('molInput').value;
                    const predType = document.getElementById('propType').value;
                    const inputMolElement = document.getElementById('input-molecule');
                    const resultMolsElement = document.getElementById('result-molecules');
                    
                    inputMolElement.innerHTML = "Processing...";
                    resultMolsElement.innerHTML = "";
                    
                    try {
                        const response = await fetch('/api/reinforce/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ 
                                smiles: smiles,
                                type: predType
                            }),
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            inputMolElement.innerHTML = `<p>Error: ${data.error}</p>`;
                            return;
                        }
                        
                        // Display input molecule
                        inputMolElement.innerHTML = `
                            <h3>Input Molecule</h3>
                            <div class="molecule-card">
                                <img src="${data.input_image}" class="molecule-img" alt="Input molecule">
                                <p>SMILES: ${data.input_smiles}</p>
                            </div>
                        `;
                        
                        // Display result molecules
                        let resultsHTML = `<h3>Results</h3>`;
                        data.results.forEach(mol => {
                            const propLabel = data.type === 'pic50' ? 'pIC50' : 'LogP';
                            const propValue = data.type === 'pic50' ? mol.pic50 : mol.logp;
                            
                            resultsHTML += `
                                <div class="molecule-card">
                                    <img src="${mol.image}" class="molecule-img" alt="Result molecule">
                                    <p>SMILES: ${mol.smiles}</p>
                                    <p>${propLabel}: ${propValue}</p>
                                </div>
                            `;
                        });
                        
                        resultMolsElement.innerHTML = resultsHTML;
                    } catch (error) {
                        console.error('Error:', error);
                        inputMolElement.innerHTML = '<p>Error processing request</p>';
                    }
                });
                
                // ViT Model Classification
                document.getElementById('vitForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const smiles = document.getElementById('vitInput').value;
                    const resultElement = document.getElementById('vit-result');
                    resultElement.innerText = "Processing...";
                    
                    try {
                        const response = await fetch('/vit/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ smiles: smiles }),
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            resultElement.innerHTML = `<p>Error: ${data.error}</p>`;
                            return;
                        }
                        
                        resultElement.innerHTML = `
                            <p><strong>SMILES:</strong> ${data.smiles}</p>
                            <p><strong>Predicted Class:</strong> ${data.predicted_class}</p>
                        `;
                    } catch (error) {
                        console.error('Error:', error);
                        resultElement.innerHTML = '<p>Error processing request</p>';
                    }
                });
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Load both models at startup
    load_segmentation_model()
    load_vit_model()
    app.run(debug=True, port=5000)