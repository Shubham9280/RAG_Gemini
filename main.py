# api.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import PyPDF2
import json
import preprocess as pre_process
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

app = FastAPI()

# Load model configuration
model_config = json.load(open("./config.json"))
Gemini_Pro_Vision_Key = model_config["gemini_pro_key"]
MODEL_CONFIG = model_config["MODEL_CONFIG"]
safety_settings = model_config["safety_settings"]

# Configure Google Generative AI model
genai.configure(api_key=Gemini_Pro_Vision_Key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=MODEL_CONFIG,
                              safety_settings=safety_settings)

@app.post("/upload_pdf/")
async def upload_pdf(username: str = Form(...), file: UploadFile = File(...)):
    # Read the uploaded PDF
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()

    # Preprocess the text
    total_number_of_paragraph, avg_single_paragraph_length, list_of_paragraph = pre_process.count_paragraphs(text)
    list_of_paragraph = [pre_process.remove_url_from_str(s_string) for s_string in list_of_paragraph]

    # Save the text in a file
    pre_process.write_in_text_file(list_of_paragraph,filename="./pdf_test.txt")

    # Save in vector database
    pre_process.save_in_vectordb("./pdf_test.txt")
    
    

    return JSONResponse(content={"message": "Text preprocessing done!"})

@app.post("/chat/")
async def chat(user_input: str = Form(...)):
    load_saved_db=FAISS.load_local("./input_data_vdb/",pre_process.load_embedding_model(),allow_dangerous_deserialization=True)
    # Retrieve relevant data from the vector database
    r_text = ""
    for s_match in pre_process.retrive_data_from_vdb(load_saved_db,user_input):
        r_text += " " + s_match.page_content

    # Prepare the input prompt for the generative model
    input_prompt = f"""Final Prompt: Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, strictly don't add anything from your side.

    Context: {r_text}
    Question: {user_input}

    Only return the helpful answer. Give direct answers with reference from the context
    """

    # Generate response using the model
    llm_response = model.generate_content(input_prompt)

    return JSONResponse(content={"response": llm_response.text})

@app.get("/")
async def read_root():
    return HTMLResponse(content=open("index.html").read())

