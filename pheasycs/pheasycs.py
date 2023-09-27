import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle
import random

nltk.download("punkt")

# Function to load intents data
def load_intents_data(filename='intents.json'):
    with open(filename) as intents_file:
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
            y_docs.append(intent['tag'])

            if intent['tag'] not in labels:
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

# Function to build and train the model
def build_and_train_model(training, output, model_filename='model.tflearn'):
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load(model_filename)
    except:
        model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
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

# Function to process user input and get responses
def get_response(model, words, labels, data, user_input):
    stemmer = LancasterStemmer()

    def bag_of_words(s):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    # Probability of correct response
    results = model.predict([bag_of_words(user_input)])

    # Picking the greatest number from probability
    results_index = np.argmax(results)

    tag = labels[results_index]

    responses = []
    for intent in data['intents']:
        if intent['tag'] == tag:
            responses.extend(intent['responses'])

    # Print the response with a typing effect
    for response in responses:
        print_with_appearance("PhEASYCS: " + response)

if __name__ == "__main__":
    intents_data = load_intents_data()
    words, labels, training, output = preprocess_data(intents_data)
    model = build_and_train_model(training, output)

    print("PhEASYCS is ready to talk!! (Type 'quit' to exit)")
    while True:
        inp = input("\nUser: ")
        if inp.lower() == 'quit':
            break

        get_response(model, words, labels, intents_data, inp)