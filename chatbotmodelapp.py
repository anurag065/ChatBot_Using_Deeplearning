import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import json

# Load the trained model and tokenizer
model = tf.keras.models.load_model('D:/spydercodes/chat_model')
with open('D:/spydercodes/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the responses from intents.json
with open('C:/Users/HP/Downloads/intents.json') as file:
    data = json.load(file)
    responses = [intent['responses'] for intent in data['intents']]

# Define the maximum sequence length
max_len = 20

# Create a function to get chatbot response
def get_chatbot_response(user_input):
    # Tokenize and pad the user input sequence
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    
    # Predict the intent label
    result = model.predict(padded_sequence)
    predicted_class = np.argmax(result)
    
    # Get the responses for the predicted class
    response = responses[predicted_class]

    # Debugging: Print the predicted class and response for inspection
    print("Predicted Class:", predicted_class)
    print("All Responses for Predicted Class:", response)
    
    # Return a random choice from the available responses
    return np.random.choice(response)

# Streamlit app
st.title("Chatbot App")
st.write("Start messaging with the bot (type 'quit' to stop)!")

user_input_list = []  # List to store user inputs

while True:
    user_input = st.text_input("User:", key=len(user_input_list))
    user_input_list.append(user_input)  # Add the user input to the list
    if user_input.lower() == "quit":
        break

    if user_input:
        chatbot_response = get_chatbot_response(user_input)
        st.text("ChatBot: " + chatbot_response)
