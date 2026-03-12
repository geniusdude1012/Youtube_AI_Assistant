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
video_id="6M5VXKLf4D4"

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
print(len(chunks))

#Step 1c & 1d : Embedding Generation and store in vector store
embeddings=OpenAIEmbeddings()
vector_store=FAISS.from_documents(chunks,embeddings)

# print(vector_store.index_to_docstore_id)

#Step 2: Retreival
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
# result=retriever.invoke("why is deep learning important for medical care")
# print(result)


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

context_text="\n\n".join(doc.page_content for doc in retreived_docs)

#Creat the final prompt from context obtained and question
final_prompt=prompt.invoke({'context':context_text,'question':question})

#Step 4: Generation



