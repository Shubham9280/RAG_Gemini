import re
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

def count_paragraphs(corpus):
    # Split the corpus into paragraphs based on newline characters
    paragraphs = corpus.split('\n\n')  # Assuming paragraphs are separated by two newline characters

    # Count the number of paragraphs
    num_paragraphs = len(paragraphs)
    
    paragraph_lengths = [len(paragraph) for paragraph in paragraphs]
    
    if paragraph_lengths:
        avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths)
    else:
        avg_paragraph_length=0
    
    return num_paragraphs,avg_paragraph_length,paragraphs

def remove_url_from_str(input_text):
    url_pattren=re.compile(r'https?://\S+|www\.\S+')
    text_without_url=url_pattren.sub(" ",input_text)
    return text_without_url

def write_in_text_file(list_paragraph):
    if os.path.exists("./pdf_test.txt"):
        with open("./pdf_test.txt", 'w') as file:
            pass
        
    for s_para in list_paragraph:
        with open('./pdf_test.txt', 'a') as f:
            f.write(s_para)
            f.write('\n')
            f.close()
    
def load_embedding_model():
    model_name="BAAI/bge-small-en-v1.5"
    model_kwargs={"device":"cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf_embedding_model
    

def save_in_vectordb(file_path):
    langchain_loader=TextLoader(file_path)
    print("***********  Converting Text into Embedding *************")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=40)
    docs = text_splitter.split_documents(langchain_loader.load())
    print("***********  Saving Embedding into Vector Database *************")
    vec_database=FAISS.from_documents(docs,load_embedding_model())
    vec_database.save_local("input_data_vdb")
    print("*************  Vector Database Created *************")

load_saved_db=FAISS.load_local("./input_data_vdb/",load_embedding_model(),allow_dangerous_deserialization=True)

def retrive_data_from_vdb(input_quetion):
    text_retriver=load_saved_db.as_retriever(search_kwargs={"k":5})
    context=text_retriver.get_relevant_documents(input_quetion)
    return context


    