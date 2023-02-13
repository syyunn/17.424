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
    
# filepath = './week2/AhAhlquist, Clayton and Levi 2014.pdf'
# filepath = './week2/Hiscox 2002.pdf'
# filepath = './week2/Mansfield and Mutz 2009.pdf'
# filepath = './week2/Osgood_Ro 2022.pdf'
# filepath = './week2/Rogowski 1987.pdf'
filepath = './week2/Scheve Slaughter 2001.pdf'

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

sources = [f'{filepath} Chunk {i}' for i in range(len(chunks))]
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
        print('\n')
        pass
    else:
        res = chain(
                {
                    "input_documents": documents,
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        print(res[:-30])
        print('\n')
        pass
    pass

# For full summary
# for doc in Documents:
#     print_answer("Summarize this section without starting your answer \'this section or this paper [verb]...\'. In addition, Include the main findings of the author as well.", [doc], show_locator=True, show_question=False)

# For specific section
for doc in Documents[11:]:
    print_answer("Summarize this section without starting your answer \'this section [verb]...\'. Include the main findings of the author as well.", [doc], show_locator=True, show_question=False)

# For specific section
# print_answer("Why does author ?", [Documents[1]])
pass