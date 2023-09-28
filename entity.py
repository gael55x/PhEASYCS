import spacy
import sys
import os

# Define the entity labels for the topics
ENTITY_LABELS = ["Effect of Temperature", "Forces", "Heat and Temperature", "Laws of Motion", "Qualitative Characteristics of Images", ""]

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python entity.py query_corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        # Pass an empty set for named_entities
        filename: tokenize(files[filename], set())
        for filename in files
    }
    
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Prepare training data with text and entity annotations
    training_data = prepare_training_data(sys.argv[1])

    # Train the custom NER model
    train_custom_ner(nlp, training_data)

    # Test the NER model
    test_custom_ner(nlp)

def prepare_training_data(corpus_dir):
    training_data = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(corpus_dir, filename)
            with open(file_path, 'r', encoding="utf8") as file:
                text = file.read()
                entities = extract_entities(text)
                training_data.append((text, {"entities": entities}))
    return training_data

def extract_entities(text):
    # Implement logic to extract entities based on synonyms and context
    # For each entity, return a tuple (start_position, end_position, entity_label)
    entities = []
    # Example: entities.append((start, end, "AI"))
    return entities

def train_custom_ner(nlp, training_data):
    # Add entity labels to the NER pipeline
    for label in ENTITY_LABELS:
        nlp.get_pipe("ner").add_label(label)

    # Disable other pipelines (e.g., tagger, parser) for training efficiency
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"]):
        optimizer = nlp.begin_training()
        for iteration in range(10):  # Adjust the number of iterations
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                example = spacy.training.example.Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], losses=losses)
            print("Iteration", iteration, "Losses:", losses)

def test_custom_ner(nlp):
    # Test the custom NER model
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text == "exit":
            break
        doc = nlp(text)
        print("Entities found:")
        for ent in doc.ents:
            print(ent.text, ent.label_)

if __name__ == "__main__":
    main()
