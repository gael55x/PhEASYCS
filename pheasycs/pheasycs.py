import sys
import time
import nltk
import tkinter as tk
from tkinter import PhotoImage
from nltk.stem.lancaster import LancasterStemmer
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

# Function to build and train the model using tf.keras
def build_and_train_model(training, output, model_filename='model.h5'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(training[0]),)),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    try:
        model = tf.keras.models.load_model(model_filename)
    except:
        model.fit(training, output, epochs=100, batch_size=8)
        model.save(model_filename)

    return model

""" Function to print text with a gradual appearance effect """
def print_with_appearance(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()  # Flush the output buffer to make the text appear immediately
        # Adjust the sleep duration to control the speed of appearance
        time.sleep(0.03)
    print()  # Move to the next line after printing the complete text


# Function to get chatbot response
def get_response(user_input):
    stemmer = LancasterStemmer()

    def bag_of_words(s):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array([bag])  # Wrap the bag in an additional array to match the expected shape

    results = model.predict(bag_of_words(user_input))  # Remove the extra list []
    results_index = np.argmax(results)
    tag = labels[results_index]

    responses = []
    for intent in intents_data['intents']:
        if 'tag' in intent and intent['tag'] == tag:
            if 'responses' in intent:  # Check if 'responses' exists in the intent
                responses.extend(intent['responses'])

    if responses:
        return ["PhEASYCS: " + responses[0]]  # Return only the first response
    else:
        return ["PhEASYCS: I'm sorry, but I don't have a response for that question. As an AI language model, I am limited by the data sets provided, which is from DEPED Physics Modules."]


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

# Load intents data, preprocess, and build the model
intents_data = load_intents_data()
words, labels, training, output = preprocess_data(intents_data)
model = build_and_train_model(training, output)

# Run the tkinter main loop
window.mainloop()
