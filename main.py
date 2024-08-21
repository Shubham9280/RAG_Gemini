import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
import re
import json
import preprocess as pre_process
import google.generativeai as genai

model_config=json.load(open("./config.json"))
Gemini_Pro_Vision_Key=model_config["gemini_pro_key"]
MODEL_CONFIG=model_config["MODEL_CONFIG"]
safety_settings=model_config["safety_settings"]

genai.configure(api_key=Gemini_Pro_Vision_Key)

model = genai.GenerativeModel(model_name = "gemini-1.5-flash",
                              generation_config = MODEL_CONFIG,
                              safety_settings = safety_settings)



# Initialize chatbot (assuming you have a chatbot library)


# Set up Streamlit layout
st.set_page_config(layout="wide")

# User input section
st.sidebar.title("Upload PDF and Username")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
# username = st.sidebar.text_input("Enter your username")



# Handle PDF file processing
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()

    # st.write(f"Uploaded by: {username}")
    st.write("Text Extracted From Input PDF")
    
#Clean Test
    total_number_of_paragraph,avg_single_paragraph_length,list_of_paragraph=pre_process.count_paragraphs(text)
    list_of_paragraph=[pre_process.remove_url_from_str(s_string)for s_string in list_of_paragraph]

    #save .txt file
    pre_process.write_in_text_file(list_of_paragraph)
    #save in vector database
    pre_process.save_in_vectordb("./pdf_test.txt")
    st.write("Text Pre-Processing Done !")

# Chatbot section
st.title("Chat with the Bot Based on Input Documents")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
# for message in st.session_state.chat_history:
#     print()
#     st.write(message)

# User input for chatbot
user_input = st.text_input("Ask something:")
if user_input:
    # response = chatbot.get_response(user_input)
    st.session_state.chat_history.append(f"You: {user_input}")
    r_text=""
    for s_match in pre_process.retrive_data_from_vdb(user_input):
        r_text=r_text+" "+s_match.page_content
    # st.write(f"{pre_process.retrive_data_from_vdb(user_input)}")
    input_prompt=f"""Final Prompt: Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, strictly don't add anything from your side.

    Context: {r_text}
    Question: {user_input}
    
    Only return the helpful answer. give direct answer with reference from the context
    """
    llm_response = model.generate_content(input_prompt)
    # st.write(f"{r_text}")
    st.session_state.chat_history.append(f"Bot: {llm_response.text}")
    st.empty()  # Clear previous output
    for message in st.session_state.chat_history:
        st.write(message)





    # Optionally, you can pass the text to the chatbot for further processing.
