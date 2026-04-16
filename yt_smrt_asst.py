import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

load_dotenv()

st.title("🎥 YouTube Video Q&A with Evaluation")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None

# Input video ID
video_id = st.text_input("YouTube Video ID", placeholder="e.g., IHZwWFHWa-w")

if video_id:
    # Load transcript button
    if st.button("Load Video"):
        with st.spinner("Loading transcript..."):
            # Get transcript
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            transcript = " ".join(chunk.text for chunk in transcript_list)
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Save to session
            st.session_state.vector_store = vector_store
            st.session_state.transcript = transcript
            st.success("✅ Video loaded!")

# Ask question
if st.session_state.vector_store:
    question = st.text_input("Ask a question about the video")
    
    if question:
        with st.spinner("Getting answer and evaluating..."):
            # Setup retriever
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Get relevant chunks
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            contexts_list = [doc.page_content for doc in docs]
            
            # Create prompt
            prompt = PromptTemplate(
                template="Context: {context}\n\nQuestion: {question}\n\nAnswer based only on context:",
                input_variables=["context", "question"]
            )
            
            # Get answer
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": question})
            answer = response.content
            
            # Display answer
            st.write("### 📝 Answer")
            st.success(answer)
            
            # ============================================
            # RAGAS EVALUATION
            # ============================================
            
            # Ground truth (you can customize based on question)
            ground_truth = """Gradient descent optimization involves:
1. Computing the gradient (direction of steepest ascent)
2. Taking a small step in the opposite direction (downhill)
3. Repeating this process iteratively until convergence"""
            
            # Prepare data for evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts_list],
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
                # Run evaluation
                evaluation_result = evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=llm,
                    embeddings=OpenAIEmbeddings(),
                )
                
                # Display metrics
                st.write("### 📊 Evaluation Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Faithfulness", 
                        f"{evaluation_result['faithfulness'][0]:.2%}",
                        help="How factually accurate the answer is based on context"
                    )
                    st.metric(
                        "Answer Relevancy", 
                        f"{evaluation_result['answer_relevancy'][0]:.2%}",
                        help="How relevant the answer is to the question"
                    )
                
                with col2:
                    st.metric(
                        "Context Precision", 
                        f"{evaluation_result['context_precision'][0]:.2%}",
                        help="How precise the retrieved context is"
                    )
                    st.metric(
                        "Context Recall", 
                        f"{evaluation_result['context_recall'][0]:.2%}",
                        help="How much relevant context was retrieved"
                    )
                
                # Show retrieved context in expander
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(doc.page_content[:300] + "...")
                        st.divider()
                
            except Exception as e:
                st.warning(f"Evaluation not available: {e}")