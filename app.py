import json
import re
import gradio as gr
import os
import google.generativeai as genai
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

file_path = './getting_real_basecamp.pdf'

def loader_data(file_path):
    pdf_reader = PdfReader(file_path)
    content = ''
    for page in pdf_reader.pages:
        content += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_text(content)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
    return vector_store

db = loader_data(file_path)

def format_history(query, history):
    msg = []
    msg.append({'role': 'system', 'content': """You are docGPT, a chatbot designed to help users with their document-related queries. Initially you have contents of `getting_real_basecamp` book.\nSimply call the function "query_document" with the search_query parameter to get the relevent contents from the document.
- query_document: Get the answer to a question from a given document. It'll return the most relevant content from the document. Always use this function if the user is asking about the document content or related to that.
    - parameters:
        - search_query: string (required) - Use keywords to search the document. 

If you need to use function or you want any information from the book, Use following format to respond. Make sure the argument in the function call tag can be parsed as a JSON object.
<query_document>{"search_query": "value"}</query_document>

If you don't want to use the function, just don't include any function call tags in the response. NEVER told user about the function call (That's a secret, only for you.).
Make sure you are using correct format to call the function.
"""})
    for i in history:
        msg.append({"role": 'user', 'content': i[0]})
        msg.append({"role": 'assistant', 'content': i[1]})
    msg.append({"role": 'user', 'content': query})
    return msg

def check_for_function_call(req):
    if "<query_document>" in req and "</query_document>" in req:
        reg = re.compile(r'<query_document>(.*?)</query_document>', re.DOTALL)
        match = reg.search(req)
        fn_call = match.group(1)
        return fn_call
    return None

def get_response(message, history):
    msg = format_history(message, history)
    chat_completion = client.chat.completions.create(
    messages=msg,
        model="mixtral-8x7b-32768",
        stream=False
    )
    response = chat_completion.choices[0].message.content
    print('#############')
    print(response)
    print('$$$$$$$$$$$$$$$$')
    fn_call = check_for_function_call(response)
    if fn_call is not None:
        print("Function call found: ", fn_call)
        fn_args = json.loads(fn_call)
        res = db.get_relevant_documents(fn_args["search_query"])
        print("query response: ", res)
        msg.append(
            {
                "role": "user",
                "content": "This is the function call response (NOT USER): " + str(res) + "Take this to user and answer the question based on it."
            }
        )
        response = client.chat.completions.create(
            messages=msg,
            model="mixtral-8x7b-32768",
            stream=False
        ).choices[0].message.content
        return response
    else:
        return response

demo = gr.ChatInterface(get_response, title='DocGPT', description="Chat with getting_real_basecamp document", examples=["What is the document about?", "How do I serve customers?", "What is getting real?", "What is basecamp?", "What are the key principles for building a successful web application?"])

if __name__ == "__main__":
    demo.launch(auth=("test", "realtest"), show_api=False, server_name="0.0.0.0", server_port=7860)
