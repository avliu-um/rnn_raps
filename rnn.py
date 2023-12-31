import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding


print(f'load data:')

embedding_matrix = np.load('data/glove_raps.npy')

train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')


print(f'embedding shape: {str(embedding_matrix.shape)}') 
print(f'data shapes: {str(train_x.shape)}, {str(train_y.shape)},{str(test_x.shape)},{str(test_y.shape)}')

model = Sequential()

# Embedding layer

vocab_size = embedding_matrix.shape[0]
model.add(
    Embedding(input_dim=vocab_size,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f'model params:')
print(model.summary())

print(f'model training:')
model.fit(train_x, train_y, epochs=20, batch_size=1, verbose=2)

model.save("my_model.keras")
