import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template # Added render_template
from collections import Counter
import re
from typing import Dict, Any

# ----------------------------------------------------------------------
# --- 1. Model Definitions (MUST match models in training files) ---
# ----------------------------------------------------------------------

# A. Emotion Detection Model Architecture (from training_test.py)
class ImprovedEmotionClassifier(nn.Module):
    """Bi-LSTM with Attention and Classification Layers."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout_prob):
        super(ImprovedEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=dropout_prob if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def attention_net(self, lstm_output):
        """Calculates attention weights and context vector."""
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_output = self.attention_net(lstm_out)
        attn_output = self.batch_norm(attn_output)
        out = self.dropout(self.relu(self.fc1(attn_output)))
        out = self.fc2(out)
        return out

# B. Music Recommendation Model Architecture (from training_music_rec.py)
class MusicRecommendationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicRecommendationModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x

# ----------------------------------------------------------------------
# --- 2. Utility Functions ---
# ----------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Simple preprocessing function (must match training preprocessing)."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    return text

def text_to_sequence(text: str, word_to_idx: Dict[str, int], max_len: int) -> torch.Tensor:
    """Converts a text string to a padded tensor sequence."""
    words = preprocess_text(text).split()
    sequence = [word_to_idx.get(word, word_to_idx.get('<UNK>', 1)) for word in words]

    # Padding/Truncating
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        padding = [word_to_idx.get('<PAD>', 0)] * (max_len - len(sequence))
        sequence.extend(padding)

    return torch.tensor([sequence], dtype=torch.long)

def get_inverse_mapping(mappings: Dict[str, Dict[Any, Any]]) -> Dict[str, Dict[Any, Any]]:
    """Generates inverse mappings for decoding encoded labels back to original strings."""
    inverse_mappings = {}
    for key, forward_map in mappings.items():
        inverse_mappings[key] = {v: k for k, v in forward_map.items()}
    return inverse_mappings

# ----------------------------------------------------------------------
# --- 3. Global Initialization and Model Loading ---
# ----------------------------------------------------------------------

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_model = None
music_rec_model = None
song_df = None # To hold our song "database"
emotion_params = {}
music_rec_mappings = {}
music_rec_inverse_mappings = {}

def load_models():
    global emotion_model, music_rec_model, song_df, emotion_params, music_rec_mappings, music_rec_inverse_mappings

    # --- Load Song Dataset ---
    SONG_DATASET_PATH = 'cleaned_music_sentiment_dataset.csv'
    try:
        print(f"Loading song dataset from {SONG_DATASET_PATH}...")
        song_df = pd.read_csv(SONG_DATASET_PATH)
        print("Song dataset loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Song dataset not found at '{SONG_DATASET_PATH}'. Please ensure the file is in the correct directory.")
        song_df = None
    except Exception as e:
        print(f"Error loading song dataset: {e}")
        song_df = None
        
    # --- Load Emotion Detection Model (Model A) ---
    EMOTION_MODEL_PATH = 'emotion_detection_pytorch_improved.pth'
    try:
        print(f"Loading Emotion Model from {EMOTION_MODEL_PATH}...")
        emotion_checkpoint = torch.load(EMOTION_MODEL_PATH, map_location=device, weights_only=False)
        emotion_model = ImprovedEmotionClassifier(
            vocab_size=emotion_checkpoint['vocab_size'],
            embedding_dim=emotion_checkpoint['embedding_dim'],
            hidden_dim=emotion_checkpoint['hidden_dim'],
            num_classes=emotion_checkpoint['num_classes'],
            num_layers=emotion_checkpoint['num_layers'],
            dropout_prob=emotion_checkpoint['dropout_prob']
        ).to(device)
        emotion_model.load_state_dict(emotion_checkpoint['model_state_dict'])
        emotion_model.eval()
        emotion_params['word_to_idx'] = emotion_checkpoint['word_to_idx']
        emotion_params['int_to_label'] = emotion_checkpoint['int_to_label']
        emotion_params['max_len'] = emotion_checkpoint['max_len']
        print("Emotion Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Emotion Model: {e}")
        emotion_model = None

    # --- Load Music Recommendation Model (Model B) ---
    MUSIC_REC_MODEL_PATH = 'music_recommendation_pytorch.pth'
    try:
        print(f"Loading Music Recommendation Model from {MUSIC_REC_MODEL_PATH}...")
        music_checkpoint = torch.load(MUSIC_REC_MODEL_PATH, map_location=device)
        music_rec_mappings = music_checkpoint['mappings']
        music_rec_inverse_mappings = get_inverse_mapping(music_rec_mappings)
        INPUT_SIZE = 4 # Hardcoded for consistency with training
        NUM_CLASSES = len(music_rec_mappings['Sentiment_Label'])
        music_rec_model = MusicRecommendationModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
        music_rec_model.load_state_dict(music_checkpoint['model_state_dict'])
        music_rec_model.eval()
        print("Music Recommendation Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Music Recommendation Model: {e}")
        music_rec_model = None

load_models() # Load models and data when the application starts

# ----------------------------------------------------------------------
# --- 4. API Endpoints ---
# ----------------------------------------------------------------------

@app.route('/')
def home():
    """Renders the frontend HTML page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Main API endpoint:
    1. Detects emotion from user text.
    2. Maps emotion to a music profile.
    3. Predicts a music sentiment based on the profile.
    4. Recommends songs from the dataset with that sentiment.
    """
    if emotion_model is None or music_rec_model is None:
        return jsonify({"error": "Models failed to load. Check console for details."}), 500

    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({"error": "No text provided for emotion detection."}), 400

    # --- STEP 1: Emotion Detection ---
    try:
        sequence = text_to_sequence(user_text, emotion_params['word_to_idx'], emotion_params['max_len']).to(device)
        with torch.no_grad():
            emotion_output = emotion_model(sequence)
            _, predicted_emotion_int = torch.max(emotion_output, 1)
        predicted_emotion_label = emotion_params['int_to_label'].get(predicted_emotion_int.item(), 'Unknown')
        if isinstance(predicted_emotion_label, np.integer):
            predicted_emotion_label = int(predicted_emotion_label)
    except Exception as e:
        print(f"Error during Emotion Detection: {e}")
        return jsonify({"error": f"Failed to run Emotion Detection: {e}"}), 500

    # --- STEP 2: Map Emotion to Music Profile ---
    emotion_to_rec_profile = {
        'joy': {'Genre': 'Pop', 'Mood': 'Happy', 'Energy': 'High', 'Danceability': 'High'},
        'sadness': {'Genre': 'Acoustic', 'Mood': 'Calm', 'Energy': 'Low', 'Danceability': 'Low'},
        'anger': {'Genre': 'Rock', 'Mood': 'Aggressive', 'Energy': 'High', 'Danceability': 'Medium'},
        'fear': {'Genre': 'Ambient', 'Mood': 'Calm', 'Energy': 'Low', 'Danceability': 'Low'},
        'love': {'Genre': 'R&B', 'Mood': 'Romantic', 'Energy': 'Medium', 'Danceability': 'Medium'},
        'surprise': {'Genre': 'EDM', 'Mood': 'Excited', 'Energy': 'High', 'Danceability': 'High'},
    }
    rec_profile = emotion_to_rec_profile.get(predicted_emotion_label, {'Genre': 'Instrumental', 'Mood': 'Neutral', 'Energy': 'Medium', 'Danceability': 'Medium'})
    
    try:
        rec_features = [music_rec_mappings[f].get(rec_profile[f], 0) for f in ['Genre', 'Mood', 'Energy', 'Danceability']]
        rec_features_tensor = torch.tensor([rec_features], dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Error during feature encoding: {e}")
        return jsonify({"error": f"Failed to encode music features: {e}"}), 500

    # --- STEP 3: Music Sentiment Prediction ---
    try:
        with torch.no_grad():
            music_output = music_rec_model(rec_features_tensor)
            _, predicted_sentiment_int = torch.max(music_output, 1)
        predicted_sentiment_label = music_rec_inverse_mappings['Sentiment_Label'].get(predicted_sentiment_int.item(), 'Unknown')
        if isinstance(predicted_sentiment_label, np.integer):
            predicted_sentiment_label = int(predicted_sentiment_label)
    except Exception as e:
        print(f"Error during Music Recommendation: {e}")
        return jsonify({"error": f"Failed to run Music Recommendation: {e}"}), 500
        
    # --- STEP 4: Song Selection ---
    recommended_songs = []
    if song_df is not None and predicted_sentiment_label != 'Unknown':
        matching_songs = song_df[song_df['Sentiment_Label'] == predicted_sentiment_label]
        if not matching_songs.empty:
            num_to_recommend = min(len(matching_songs), 5)
            recommended_songs_df = matching_songs.sample(n=num_to_recommend)
            recommended_songs = recommended_songs_df[['Song_Title', 'Artist']].to_dict(orient='records')

    # --- Final Response ---
    if recommended_songs:
        final_recommendation = f"Based on your detected emotion, here are some songs with a '{predicted_sentiment_label}' vibe:"
    else:
        final_recommendation = f"Based on your emotion, we recommend music with a '{predicted_sentiment_label}' sentiment. Add songs to the dataset to get specific recommendations."
    
    return jsonify({
        "input_text": user_text,
        "detected_emotion": predicted_emotion_label,
        "recommended_music_sentiment": predicted_sentiment_label,
        "final_recommendation": final_recommendation,
        "recommended_songs": recommended_songs
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)

