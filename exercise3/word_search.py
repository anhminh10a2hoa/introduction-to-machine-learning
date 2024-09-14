import numpy as np

# Load word embeddings with UTF-8 encoding
def load_embeddings(file_path):
    words = []
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            words.append(word)
            vectors.append(vector)
    return words, np.array(vectors)

# Function to find the most similar words
def find_similar_words(word, words, vectors, top_n=3):
    if word not in words:
        print(f"'{word}' not found in vocabulary.")
        return []
    
    word_index = words.index(word)
    word_vector = vectors[word_index]
    
    # Compute Euclidean distances
    distances = np.linalg.norm(vectors - word_vector, axis=1)
    
    # Get the indices of the closest words (including the word itself)
    closest_indices = np.argsort(distances)[:top_n]
    return [words[idx] for idx in closest_indices]

if __name__ == "__main__":
    # Load embeddings
    words, vectors = load_embeddings('word_embeddings.txt')

    # Test words
    test_words = ["king", "europe", "frog"]
    
    for word in test_words:
        similar_words = find_similar_words(word, words, vectors)
        print(f"Similar words to '{word}': {similar_words}")
