import os
import pickle
from data_preprocessing import preprocess_data
from model import build_model

# --- Configuration ---
DATA_FILE_PATH = os.path.join('data', 'faqs.txt')
MODEL_SAVE_PATH = os.path.join('saved_models', 'faq_model.h5')
TOKENIZER_SAVE_PATH = os.path.join('saved_models', 'tokenizer.pkl')

# --- Main Training Logic ---
if __name__ == "__main__":
    print("Step 1: Preprocessing data...")
    X, y, tokenizer, max_len, total_words = preprocess_data(DATA_FILE_PATH)
    
    print("\nData processed:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Total unique words: {total_words}")
    print(f"Max sequence length: {max_len}")
    
    print("\nStep 2: Building the model...")
    model = build_model(total_words, max_len)
    
    print("\nStep 3: Training the model...")
    model.fit(X, y, epochs=100, verbose=1)
    
    print("\nStep 4: Saving the model and tokenizer...")
    model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump((tokenizer, max_len), f)
        
    print("\nTraining complete! Model and tokenizer are saved in the 'saved_models' folder.")