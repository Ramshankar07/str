# Import modules
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Ram07/emp1_dialog")
tokenizer = AutoTokenizer.from_pretrained("Ram07/emp1_dialog")

# Create list of messages
messages = [
        {
        "role": "system",
        "content": "You are a friendly empathetic chatbot who always responds in the style of a therapist",
    },
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, I am a XYz from "},
 
  
     {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I've been practicing and preparing as much as I can, but I can't shake off the nerves. I keep doubting whether I'm good enough or if I'll make a good impression.!"},
 ]
# Apply chat template
messages = tokenizer.apply_chat_template(messages)

# Create conversational pipeline
pipe = pipeline("conversational", model=model, tokenizer=tokenizer)

# Create streamlit app
st.title("Chatbot with Streamlit")
st.write("This is an example of how to create a chatbot with streamlit using transformer's conversational pipeline.")

# Display user input widget
user_input = st.text_input("Enter your message:")

# Display output widget if user clicks button
if st.button("Generate"):
    # Generate response from pipe object
    response = pipe(user_input)
    # Display response in markdown format
    st.markdown(response)
