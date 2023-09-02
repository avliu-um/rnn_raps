from keras.preprocessing.text import Tokenizer
import numpy as np
import random
import os


def get_verses(raps_file_path='./source/rap_lyrics.txt'):
    # Get verses
    verses = []
    current_verse = []

    with open(raps_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line:  # Non-empty line
                current_verse.extend(line.split())
            elif current_verse:  # Empty line, but we have words in the current verse
                verses.append(current_verse)
                current_verse = []

        if current_verse:  # If there's a verse left after reading the file
            verses.append(current_verse)

            
    # Shuffle the verses here so that train/test is independent of rapper 
    random.shuffle(verses)

    return verses


def tokenize(texts):

    # Tokenize texts (words --> numbers)
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    tok_seqs = tokenizer.texts_to_sequences(texts)

    return tok_seqs, tokenizer 


def gen_rnn_xy(sequences, vocab_size, x_length=50):

    # Split data into RNN-style splits 
    # (words 1-50 predict word 51, words 2-51 predict word 52, etc.)
    features = []
    labels = []

    # Iterate through the sequences of tokens (for us, verses)
    for seq in sequences:

        # Create multiple training examples from each sequence
        for i in range(x_length, len(seq)):
            
            # Extract the features and label
            extract = seq[i - x_length:i + 1]

            # Set the features and label
            features.append(extract[:-1])
            labels.append(extract[-1])
            
    features = np.array(features)

    # One-hot the labels (i.e. Y)
    label_array = np.zeros((len(features), vocab_size), dtype=np.int8)

    for example_index, word_index in enumerate(labels):
        label_array[example_index, word_index] = 1

    return features, label_array 


def split_train_test(x,y):
    # Split into train/test
    n = x.shape[0] 
    train_ratio = 0.8
    split = int(n*train_ratio)

    train_x = x[:split]
    train_y = y[:split]

    test_x = x[split:]
    test_y = y[split:]

    return train_x, train_y, test_x, test_y


def gen_embedding_matrix(vocabulary, vocab_size, emb_file='./source/glove.6B.100d.txt'):
    # Pre-trained embeddings
    # Load in embeddings
    glove = np.loadtxt(emb_file, dtype='str', comments=None)

    # Extract the vectors and words
    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]

    # Create lookup of words to vectors
    word_lookup = {word: vector for word, vector in zip(words, vectors)}

    # New matrix to hold word embeddings
    embedding_matrix = np.zeros((vocab_size, vectors.shape[1]))

    for i, word in enumerate(vocabulary):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
            
    return embedding_matrix

def main():

    verses = get_verses()
    print(f'collected {len(verses)} verses')

    tok_verses, tok = tokenize(verses)
    vocab = tok.word_index.keys()
    vocab_size = len(vocab)+1
    print(f'tokenized verses with {vocab_size}-word long vocabulary')

    if not os.path.isfile('data/glove_raps.npy'):
        embedding_matrix = gen_embedding_matrix(vocab, vocab_size)
        np.save('data/glove_raps.npy', embedding_matrix)
        print(f'generated embedding matrix of size {str(embedding_matrix.shape)}')

    x, y = gen_rnn_xy(tok_verses, vocab_size)
    train_x, train_y, test_x, test_y = split_train_test(x, y)
    print(f'data shapes: {str(train_x.shape)}, {str(train_y.shape)},{str(test_x.shape)},{str(test_y.shape)}')

    np.save('data/train_x.npy', train_x) 
    np.save('data/train_y.npy', train_y) 
    np.save('data/test_x.npy', test_x) 
    np.save('data/test_y.npy', test_y) 

if __name__=='__main__':
    main()
