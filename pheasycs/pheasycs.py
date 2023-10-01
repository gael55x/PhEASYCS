import sys
import time
import nltk
import tkinter as tk
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
import json
import random

nltk.download("punkt")

# Function to load intents data
def load_intents_data(filename='intents.json'):
    with open(filename, 'r', encoding='utf-8') as intents_file:
        return json.load(intents_file)

# Function to preprocess data
def preprocess_data(data):
    stemmer = LancasterStemmer()
    words = []
    labels = []
    x_docs = []
    y_docs = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            x_docs.append(wrds)
            # Use the get method to provide a default value ('unknown') when 'tag' is missing
            if 'tag' in intent:
                y_docs.append(intent['tag'])
            else:
                y_docs.append('unknown')

            if 'tag' in intent and intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(x_docs):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(y_docs[x])] = 1

        training.append(bag)
        output.append(output_row)

    return words, labels, np.array(training), np.array(output)

"""
    Build and train a deep learning model using TensorFlow/Keras.

    This function creates a sequential neural network model with configurable layers,
    compiles it with specified optimizer and loss, and optionally loads a pre-trained model
    or trains a new one based on provided training data.

    Args:
        training (numpy.ndarray): The input training data.
        output (numpy.ndarray): The output or target training data.
        model_filename (str): The name of the file to save/load the trained model (default is 'model.h5').

    Returns:
        tf.keras.Model: The trained deep learning model.

    Raises:
        IOError: If loading a pre-trained model fails.

    Example:
        # Build and train a model
        model = build_and_train_model(training_data, target_data)

"""
def build_and_train_model(training, output, model_filename='model.h5'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(training[0]),)),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print the number of neurons in the input and output layers
    num_input_neurons = len(training[0])
    num_output_neurons = len(output[0])

    print(f"Number of neurons in the input layer: {num_input_neurons}")
    print(f"Number of neurons in the output layer: {num_output_neurons}")
    
    try:
        model = tf.keras.models.load_model(model_filename)
    except:
        model.fit(training, output, epochs=100, batch_size=8)
        model.save(model_filename)

    """
    Model: "sequential"
    __________________________________________________________________
     Layer (type)                Output Shape              Param #
    ==================================================================
     dense (Dense)               (None, 8)                 5112

     dense_1 (Dense)             (None, 8)                 72

     dense_2 (Dense)             (None, 222)               1998

    ==================================================================
    Total params: 7182 (28.05 KB)
    Trainable params: 7182 (28.05 KB)
    Non-trainable params: 0 (0.00 Byte)
    __________________________________________________________________
    dense True
    [TensorShape([638, 8]), TensorShape([8])]
    dense_1 True
    [TensorShape([8, 8]), TensorShape([8])]
    dense_2 True
    [TensorShape([8, 222]), TensorShape([222])]
    """

    # Display a summary of the model architecture, including the number of parameters
    model.summary()

    # Access the details of each layer
    for layer in model.layers:
        print(layer.name, layer.trainable)
        if hasattr(layer, 'weights'):
            print([w.shape for w in layer.weights])

    return model

""" Function to print text with a gradual appearance effect """
def print_with_appearance(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()  # Flush the output buffer to make the text appear immediately
        # Adjust the sleep duration to control the speed of appearance
        time.sleep(0.03)
    print()  # Move to the next line after printing the complete text


# Load intents data, preprocess, and build the model
intents_data = load_intents_data()
words, labels, training, output = preprocess_data(intents_data)
model = build_and_train_model(training, output)
# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Preprocess and vectorize your data during initialization
corpus = [intent['patterns'] for intent in intents_data['intents']]
X = vectorizer.fit_transform([' '.join(pattern) for pattern in corpus])
y = np.array([intent.get('tag', 'unknown') for intent in intents_data['intents']])


def get_response(user_input, confidence_threshold=0.50):
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and patterns
    similarity_scores = cosine_similarity(user_vector, X)

    # Find the intent with the highest similarity score
    max_similarity_index = np.argmax(similarity_scores)
    max_similarity = similarity_scores[0, max_similarity_index]

    if max_similarity < confidence_threshold:
        return ["PhEASYCS: I'm sorry, but I don't have a response for that question. As an AI language model, I am limited by my own data sets, which is from DEPED Physics modules. I am only designed to teach anyone Physics."]

    tag = y[max_similarity_index]

    responses = []
    for intent in intents_data['intents']:
        if 'tag' in intent and intent['tag'] == tag:
            if 'responses' in intent:  # Check if 'responses' exists in the intent
                responses.extend(intent['responses'])

    if responses:
        return ["PhEASYCS: " + responses[0]]  # Return only the first response
    else:
        return ["PhEASYCS: I'm sorry, but I don't have a response for that question. As an AI language model, I am limited by my own data sets, which is from DEPED Physics modules. I am only designed to teach anyone Physics. "]
    
# Create a tkinter window
window = tk.Tk()
window.title("PhEASYCS Chatbot")
window.geometry("800x600")  # Set a larger window size

# Create a frame for the user input and chat display
frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)  # Expand to fill the window

# Set background colors for the user and chatbot sections
user_bg_color = "#303030"  # Dark background color for the user section
chatbot_bg_color = "#212121"  # Slightly darker background color for the chatbot section

# Create a text box for displaying chat with the user's background color
user_chat_display = tk.Text(frame, height=20, width=30)
user_chat_display.config(bg=user_bg_color, fg="white", padx=10, pady=10, font=("Arial", 12))  # Customize user section appearance
user_chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a text box for displaying chat with the chatbot's background color
chatbot_chat_display = tk.Text(frame, height=20, width=70)
chatbot_chat_display.config(bg=chatbot_bg_color, fg="white", padx=10, pady=10, font=("Arial", 12))  # Customize chatbot section appearance
chatbot_chat_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create an entry field for user input
user_input_entry = tk.Entry(window, width=50, font=("Arial", 12))
user_input_entry.pack()

# Function to handle user input from the GUI
def handle_user_input():
    user_input = user_input_entry.get()
    if user_input.lower() == 'quit':
        window.quit()
        return

    # Display the user's query in the chat display with a typing effect
    user_query_display = "User: " + user_input + "\n"
    print_with_appearance(user_query_display)
    user_chat_display.insert(tk.END, user_query_display)

    responses = get_response(user_input)

    # Display the chatbot's responses with a typing effect in the chat display
    for response in responses:
        for char in response:
            chatbot_chat_display.insert(tk.END, char)
            chatbot_chat_display.update_idletasks()  # Update the display immediately
            time.sleep(0.03)  # Adjust the sleep duration for typing effect

        chatbot_chat_display.insert(tk.END, "\n")  # Move to the next line after each response

    # Clear the user input field
    user_input_entry.delete(0, tk.END)

# Create a button to send user input
send_button = tk.Button(window, text="Send", command=handle_user_input, font=("Arial", 12))
send_button.pack()

print("PhEASYCS is ready to talk!! (Type 'quit' to exit)")

# Run the tkinter main loop
window.mainloop()
