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


