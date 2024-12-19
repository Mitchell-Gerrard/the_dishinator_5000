import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load meal descriptions from JSON file
with open('the_dishernator_5000/food_descriptions.json', 'r') as file:
    meals = json.load(file)

# Step 1: Combine descriptions for each meal
combined_descriptions = []
labels = []

# Iterate over each meal and its descriptions
for meal_idx, (meal, descriptions) in enumerate(meals.items()):
    # Combine all descriptions into one string
    combined_text = ' '.join(descriptions)
    combined_descriptions.append(combined_text)
    # Assign a label for the meal (using the index as the label)
    labels.append(meal_idx)

# Step 2: Tokenize and pad combined descriptions
# Initialize the tokenizer
tokenizer = Tokenizer()
# Fit the tokenizer on the combined descriptions
tokenizer.fit_on_texts(combined_descriptions)

# Convert combined descriptions into sequences of integers
sequences = tokenizer.texts_to_sequences(combined_descriptions)

# Determine the maximum sequence length for padding
max_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform shape (post-padding with zeros)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to numpy array for model training
labels = np.array(labels, dtype=np.int32)

# Step 3: Build the model
# Define the vocabulary size based on the tokenizer
vocab_size = len(tokenizer.word_index) + 1
# Define the embedding dimension
embedding_dim = 16

# Build a Sequential model
model = Sequential([
    # Embedding layer to convert integer sequences to dense vectors
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    # Global average pooling layer to reduce the dimensionality
    GlobalAveragePooling1D(),
    # Dense layer with ReLU activation
    Dense(16, activation='relu'),
    # Output layer with softmax activation (number of units = number of meals)
    Dense(len(meals), activation='softmax')
])

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
# Train the model on the padded sequences and labels
model.fit(padded_sequences, labels, epochs=300, verbose=1)

# Step 5: Function to find the best match for a new description
def find_best_match_meal(description):
    # Combine the new description into a single text string
    combined_text = ' '.join(description)
    
    # Convert the combined text to a sequence of integers
    sequence = tokenizer.texts_to_sequences([combined_text])
    # Pad the sequence to match the maximum length
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Predict probabilities for each meal
    predictions = model.predict(padded_sequence)
    # Create a dictionary of meal probabilities
    meal_probabilities = {meal: predictions[0][idx] for idx, meal in enumerate(meals.keys())}
    
    # Sort the meals by probability in descending order
    sorted_meals = sorted(meal_probabilities.items(), key=lambda item: item[1], reverse=True)
    
    # Truncate the result to the top 5 matches
    top_5_meals = sorted_meals[:5]
    
    # Format the result as a list of strings with meal names and match percentages
    return [f"{meal}: {prob*100:.2f}% match" for meal, prob in top_5_meals]

# Example usage
new_description = ["savoury"]
best_matches = find_best_match_meal(new_description)
print(f"Best matches for '{new_description}': {best_matches}")