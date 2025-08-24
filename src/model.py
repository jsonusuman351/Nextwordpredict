from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(total_words, max_len):
    """Builds and compiles the LSTM model."""
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_len - 1))
    model.add(LSTM(150)) # You can experiment with adding another LSTM layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model