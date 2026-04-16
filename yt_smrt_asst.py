from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# ============================================
# HUGGING FACE SETUP (UPDATED)
# ============================================

# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# if not HUGGINGFACE_API_TOKEN:
#     print(" Warning: HUGGINGFACE_API_TOKEN not found in .env file")
#     print("Please add: HUGGINGFACE_API_TOKEN=your_token_here")

# #  FIXED: Use HuggingFaceHub instead of HuggingFaceEndpoint
# model = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-large",
#     task="text-generation",
#     max_new_tokens=100
# )

# model = ChatHuggingFace(model=model)

# Alternative: Use ChatHuggingFace for chat models
# from langchain_huggingface import ChatHuggingFace
# model = ChatHuggingFace(
#     model_id="mistralai/Mistral-7B-Instruct-v0.3",
#     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
# )
model=ChatOpenAI()
# Embeddings (works the same)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)



# ============================================
# STEP 1: LOAD AND PROCESS YOUTUBE TRANSCRIPT
# ============================================

video_id = "IHZwWFHWa-w"
ytt_api = YouTubeTranscriptApi()

try:
    transcript_list = ytt_api.fetch(video_id, languages=["en"])
    transcript = "".join(chunk.text for chunk in transcript_list)
    print(f"\n Transcript loaded! Length: {len(transcript)} characters")
except TranscriptsDisabled:
    print(" No captions available for this video")
    exit()
except Exception as e:
    print(f" Error fetching transcript: {e}")
    exit()

# ============================================
# STEP 2: TEXT SPLITTING
# ============================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.create_documents([transcript])
print(f" Split into {len(chunks)} chunks")

# ============================================
# STEP 3: CREATE VECTOR STORE
# ============================================

vector_store = FAISS.from_documents(chunks, embeddings)
print(" Vector store created with FAISS")

# ============================================
# STEP 4: SETUP RETRIEVER
# ============================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ============================================
# STEP 5: CREATE PROMPT TEMPLATE
# ============================================

prompt = PromptTemplate(
    template="""<s>[INST] <<SYS>>
You are a helpful assistant that answers questions based on the provided context.
<</SYS>>

Context information:
{context}

Question: {question}

Instructions:
- Answer only using the information from the context above
- If the context doesn't contain the answer, say "I don't have enough information"
- Keep your answer concise and accurate

Answer: [/INST]""",
    input_variables=['context', 'question']
)

# ============================================
# STEP 6: CREATE RAG CHAIN
# ============================================

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Create parallel chain for context and question
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# Complete RAG chain
main_chain = parallel_chain | prompt | model | StrOutputParser()

# ============================================
# STEP 7: ASK A QUESTION
# ============================================

question = "how to optimize gradient descent?"

print(f"\n" + "="*50)
print(f"QUESTION: {question}")
print("="*50)

try:
    # Get answer
    answer = main_chain.invoke(question)
    print(f"\n ANSWER:\n{answer}")
    
    # Get retrieved documents for evaluation
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    
except Exception as e:
    print(f" Error during inference: {e}")
    print("\nTrying alternative model...")
    
    # Fallback to a smaller, more reliable model
    
    main_chain = parallel_chain | prompt | model | StrOutputParser()
    answer = main_chain.invoke(question)
    print(f"\n ANSWER (using flan-t5-large):\n{answer}")
    
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

# ============================================
# STEP 8: RAGAS EVALUATION
# ============================================

print("\n" + "="*50)
print("RAGAS EVALUATION")
print("="*50)

ground_truth = """Gradient descent optimization involves:
1. Computing the gradient (direction of steepest ascent)
2. Taking a small step in the opposite direction (downhill)
3. Repeating this process iteratively until convergence
The step size is controlled by the learning rate."""

data = {
    "question": [question],
    "answer": [answer],
    "contexts": [contexts],
    "ground_truth": [ground_truth]
}

dataset = Dataset.from_dict(data)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

try:
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=model,
        embeddings=embeddings,
    )
    
    print("\n EVALUATION RESULTS:")
    print("-" * 30)
    print(f"Faithfulness: {evaluation_result['faithfulness'][0]:.4f}")
    print(f"Answer Relevancy: {evaluation_result['answer_relevancy'][0]:.4f}")
    print(f"Context Precision: {evaluation_result['context_precision'][0]:.4f}")
    print(f"Context Recall: {evaluation_result['context_recall'][0]:.4f}")
except Exception as e:
    print(f"\n⚠️ Evaluation error: {e}")