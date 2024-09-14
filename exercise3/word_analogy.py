import numpy as np

# Load word embeddings with UTF-8 encoding
def load_embeddings(file_path):
    words = []
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:  # Added 'utf-8' encoding
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            words.append(word)
            vectors.append(vector)
    return words, np.array(vectors)

# Function to find the closest word for a given analogy
def find_analogy(word1, word2, word3, words, vectors):
    if word1 not in words or word2 not in words or word3 not in words:
        print("One or more words not found in vocabulary.")
        return []

    idx1, idx2, idx3 = words.index(word1), words.index(word2), words.index(word3)
    vector1, vector2, vector3 = vectors[idx1], vectors[idx2], vectors[idx3]

    # Analogy vector: (word2 - word1) + word3
    analogy_vector = vector3 + (vector2 - vector1)

    # Compute Euclidean distances to find closest word
    distances = np.linalg.norm(vectors - analogy_vector, axis=1)
    closest_indices = np.argsort(distances)[:2]  # Return two closest matches
    return [words[idx] for idx in closest_indices if idx not in [idx1, idx2, idx3]]

if __name__ == "__main__":
    # Load embeddings
    words, vectors = load_embeddings('word_embeddings.txt')

    # Test analogies
    analogies = [
        ("king", "queen", "prince"),
        ("finland", "helsinki", "china"),
        ("love", "kiss", "hate")
    ]

    for word1, word2, word3 in analogies:
        analogy_result = find_analogy(word1, word2, word3, words, vectors)
        print(f"'{word1}' is to '{word2}' as '{word3}' is to {analogy_result}")
