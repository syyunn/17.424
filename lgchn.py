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

filepath = 'clemente.pdf'

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

sources = [f'{filepath} 1987 Chunk {i}' for i in range(len(chunks))]
Documents = make_documents(chunks, sources)

import dotenv
import os
dotenv.load_dotenv(".env")
openai_apikey = os.getenv("OPENAI_API_KEY")

chain = load_qa_with_sources_chain(OpenAI(temperature=0, api_key=openai_apikey))

def print_answer(question, documents, show_locator=False, show_question=False):
    if show_question:
        print(question)
    if show_locator:
        print(
        chain(
            {
                "input_documents": documents,
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
        )
    else:
        res = chain(
                {
                    "input_documents": documents,
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        print(res[:-30])

## For full summary
# for doc in Documents:
#     print_answer("Summarize this section without starting your sentence \'this section ... \'", [doc], show_locator=True, show_question=False)

## For specific section
print_answer("What is the meaning of environmental crackdown in this paper?", [Documents[1]])