import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load saved model and tokenizer ---
MODEL_PATH = os.path.join('saved_models', 'faq_model.h5')
TOKENIZER_PATH = os.path.join('saved_models', 'tokenizer.pkl')

print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer, max_len = pickle.load(f)
print("Model and tokenizer loaded successfully.")

def generate_text(seed_text, num_words_to_generate):
    """Generates text based on a seed phrase."""
    output_text = seed_text
    for _ in range(num_words_to_generate):
        # Tokenize and pad the current text
        token_text = tokenizer.texts_to_sequences([output_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
        
        # Predict the next word's index
        predicted_index = np.argmax(model.predict(padded_token_text, verbose=0))
        
        # Find the word corresponding to the index
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break
        
        if not next_word:
            break
            
        output_text += " " + next_word
        
    return output_text

if __name__ == "__main__":
    # Example usage
    seed_text = "what is the duration"
    generated_sequence = generate_text(seed_text, 10) # Generate 10 more words
    
    print("-" * 50)
    print(f"Seed Text: '{seed_text}'")
    print(f"Generated Text: '{generated_sequence}'")
    print("-" * 50)