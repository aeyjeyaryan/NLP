#importing necessary libraries
import os
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#Reading the data
import json
with open(r'C:\Users\sahil\OneDrive\Desktop\intents.json', 'r') as file:
    data = json.load(file)
intents = data['intents']


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
counter = 0

def main():
    print("Welcome To THE APP NAME!")
    print("Hi, Feel free to talk about your issues!")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Thank you for chatting with me. Have a great day!")
            break

        response = chatbot(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()

