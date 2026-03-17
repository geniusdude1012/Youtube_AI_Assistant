import yt_smrt_asst

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


#Load DotEnv
load_dotenv()

#Create a model
model=ChatOpenAI()

#Create an embedding model
embeddings=OpenAIEmbeddings()

#Step 1a: Indexing(Data Ingestion)
video_id="ukzFI9rgwfU"

ytt_api=YouTubeTranscriptApi()

try:
    
    transcript_list=ytt_api.fetch(video_id,languages=["en"])
    
    #Convert into plain text
    transcript="".join(chunk.text for chunk in transcript_list)
    print(transcript)

    
except TranscriptsDisabled:
    print("No captions available for this video")
    
#Step 1b: Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents(transcript)
# print(len(chunks))

#Step 1c & 1d : Embedding Generation and store in vector store
embeddings=OpenAIEmbeddings()
vector_store=FAISS.from_documents(chunks,embeddings)

# print(vector_store.index_to_docstore_id)

#Step 2: Retreival
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
result=retriever.invoke("what is machine learning")
print(result)


#Step 3: Augmentation
prompt=PromptTemplate(
    template="""
    You are an helpful assistant.
    Answer only from provided transcript context.
    
    If the context is insufficent, just say you don't know.
    
    {context}
    Question:{question}
    """ ,
    input_variables=['context','question']
)

question="why is deep learning important for medical care"
retreived_docs=retriever.invoke(question)

# context_text="\n\n".join(doc.page_content for doc in retreived_docs)

# #Creat the final prompt from context obtained and question
# final_prompt=prompt.invoke({'context':context_text,'question':question})

# #Step 4: Generation
# response=model.invoke(final_prompt)
# print(response)

#Building a chain
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retreived_docs):
    context_text="\n\n".join(doc.page_content for doc in retreived_docs)
    return context_text

#Creating parallel chain for query and context
parallel_chains=RunnableParallel({
    'context': retriever| RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

# parallel_chains.invoke("What is the main topic?")

parser=StrOutputParser()

main_chain=parallel_chains | prompt | model | parser

result=main_chain.invoke("what is machine learning?")
print(result)

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

# Import your functions if modularized
# from your_file import format_docs


def test_format_docs():
    docs = [
        Document(page_content="Machine learning is AI"),
        Document(page_content="Deep learning is subset")
    ]

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    result = format_docs(docs)

    assert "Machine learning" in result
    assert "Deep learning" in result
    
@patch("langchain_openai.ChatOpenAI")    
def test_llm_chain(mock_llm):

    mock_model = MagicMock()
    mock_model.invoke.return_value = "Machine learning is a field of AI."

    mock_llm.return_value = mock_model

    response = mock_model.invoke("What is ML?")

    assert "Machine learning" in response