from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

import PyPDF2

def extract_text_from_pdf(filepath):
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        text = ''
        for page_number in range(number_of_pages):
            page = reader.pages[page_number]
            text += page.extract_text()
        return text

filepath = 'week2/Rogowski 1987.pdf'

maxlen = 4097

def spliter(text, maxlen):
    chunks = []
    for i in range(0, len(text), maxlen):
        chunks.append(text[i:i+maxlen])
    return chunks

text = extract_text_from_pdf(filepath)
chunks = spliter(text, maxlen)
print(len(text))

def make_documents(chunks, sources):
    Documents = []
    for chunk, source in zip(chunks, sources):
        Documents.append(Document(page_content=chunk, metadata={"source": source}))
    return Documents

sources = [f'Rogowski 1987 {i}' for i in range(len(chunks))]
Documents = make_documents(chunks, sources)

import dotenv
import os
dotenv.load_dotenv(".env")
openai_apikey = os.getenv("OPENAI_API_KEY")

chain = load_qa_with_sources_chain(OpenAI(temperature=0, api_key=openai_apikey))

def print_answer(question):
    print(
        chain(
            {
                "input_documents": [Documents[0]],
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )

print_answer("What is The Stolper-Samuelson Theorem explained in this paper?")
