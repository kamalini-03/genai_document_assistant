import streamlit as st # For building the web application.
from pypdf import PdfReader # For PDF text extraction.
import re # For cleaning and preprocessing the Text.
from collections import Counter # For detecting repeated headers in documents.
from sentence_transformers import SentenceTransformer # For generating vector embeddings for semantic search.
import numpy as np # For numerical operations and converting embeddings to FAISS-compatible format.
import faiss # Used as a vector database for similarity search.
import os # For accessing environment variables such as API keys.
from dotenv import load_dotenv # For loading environment variables from the .env file.
import google.generativeai as genai
# Using Gemini API for answer generation because it provides a free tier, 
# high-quality responses, and is easy to integrate with RAG applications.

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-3.5-flash")

# Using SentenceTransformer ("all-MiniLM-L6-v2"), a lightweight Hugging Face embedding model 
# chosen because it runs locally, is free to use, does not require API keys,
# and provides good semantic search performance for RAG systems.

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Creating the title and description for the Streamlit application.
st.title("Generative AI-Powered Document Assistant")
st.write("Upload your documents and ask questions about them.")

# Creating chat history storage for follow-up question handling.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Allowing users to upload multiple PDF and TXT documents.
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Checking whether the user uploaded any files or not.
if uploaded_files:
    st.write(f"{len(uploaded_files)} files uploaded successfully.")

    # Storing chunks from all uploaded files.
    chunks = []
    chunk_id = 1
    chunk_size = 500
    overlap = 50
    

    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")

        # Processing PDF Files.
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            st.write(f"Total pages: {len(pdf_reader.pages)}")
    
            # Storing extracted text and metadata for each page.
            pages_text = []
            for page_number, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()

                if page_text:
                    pages_text.append({
                        "file_name": uploaded_file.name,
                        "page_number": page_number + 1,
                        "text": page_text
                    })
                    
            st.write(f"{len(pages_text)} pages extracted successfully.")

            # Detecting if there are any repeated headers.
            candidate_headers = []
            for page in pages_text:
                lines = [
                    line.strip()
                    for line in page["text"].split("\n")
                    if line.strip()
                ]

                candidate_headers.extend(lines[:3])
            header_counts = Counter(candidate_headers)
            header_threshold = max(2, len(pages_text)*0.5)
            detected_headers = [
                line
                for line, count in header_counts.items()
                if count >= header_threshold
            ]
            
            # Storing cleaned pages after noise removal.
            cleaned_pages = []
            
            # Removing empty lines, page numbers and repeated headers.
            for page in pages_text:
                lines = page["text"].split("\n")
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line in detected_headers:
                        continue
                    if line.isdigit():
                        continue
                    filtered_lines.append(line)
                    
                page_clean_text = "\n".join(filtered_lines)
                    
                cleaned_text = re.sub(r"\s+", " ", page_clean_text).strip()
                cleaned_pages.append({
                    "file_name": page["file_name"],
                    "page_number": page["page_number"],
                    "text": cleaned_text
                })
            
            # Updating pages_text with cleaned content.
            pages_text = cleaned_pages
        
        # Processing Txt Files.
        elif uploaded_file.type == "text/plain":

            # Reading text file content and decoding it into readable UTF-8 text format.
            document_text = uploaded_file.read().decode("utf-8")
            document_text = re.sub(r"\s+", " ", document_text).strip()
            pages_text = [{
                "file_name": uploaded_file.name,
                "page_number": 1,
                "text": document_text

            }]

        # Performing fixed-size chunking with overlap.
        for page in pages_text:
            words = page["text"].split()
            start = 0
            while start < len(words):
                end = start + chunk_size

                # Creating a chunk from the selected word range.
                chunk_text = " ".join(words[start:end])
                
                # Storing chunk text along with metadata.
                chunks.append({
                    "file_name": page["file_name"],
                    "page_number": page["page_number"],
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })
                chunk_id += 1

                # Moving to the next chunk while preserving overlap between consecutive chunks.
                start += chunk_size - overlap

        # Displaying the total number of chunks created.
        st.write(f"{len(chunks)} chunks created.")
        
        # Displaying a preview of the first chunk for verification.
        if chunks:
            st.write("First chunk preview:")
            st.write(chunks[0]["text"][:500])

        st.write("Document processing completed successfully.")

    # Extracted only the chunk text for embedding generation
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]
    # Converting chunks into vector embeddings
    embeddings = embedding_model.encode(chunk_texts)
    # Displaying the information of embeddings
    st.write(f"{len(embeddings)} embeddings created.")
    st.write(f"Embedding dimension: {embeddings.shape[1]}")


    # Using FAISS as the vector database because it is lightweight,
    # runs locally, provides fast similarity search.

    # FAISS only accepts NumPy arrays in float32 format. 
    # So we are converting embeddings to float32 format.      
    embeddings = np.array(embeddings).astype("float32")
    
    # Get the embedding dimension, i.e the size of each embedding vector.
    dimension = embeddings.shape[1]
    
    # Create a FAISS vector database index using Euclidean distance (L2 distance) for similarity search.
    index = faiss.IndexFlatL2(dimension)
    
    # Store all chunk embeddings inside FAISS index.
    index.add(embeddings)
    
    # Display the vector database information, i.e the number of vectors stored.
    st.write(f"FAISS index contains {index.ntotal} vectors.")

    # Creating a chat input box for user questions.
    user_query = st.chat_input("Ask a question about the uploaded documents")

    # Checking whether the user entered a question or not, if entered then convert it into embedding vector.
    if user_query:
        # By default, use the current user question for retrieval.
        query_for_search = user_query
        # If previous conversation exists, check whether the current
        # question depends on the previous question.
        if st.session_state.chat_history:
            previous_question = (
                st.session_state.chat_history[-1]["question"]
            )

            rewrite_prompt = f"""
        Previous Question:
        {previous_question}

        Current Question:
        {user_query}

        If the current question depends on the previous question, rewrite it as a complete standalone question.

        If the current question is unrelated to the previous question, return it unchanged.

        Return only the rewritten question.
        """

            query_for_search = (llm_model.generate_content(rewrite_prompt).text.strip())

        # Generating embedding for the query that will be used for retrieval.
        query_embedding = embedding_model.encode([query_for_search])
        query_embedding = np.array(query_embedding).astype("float32")

        st.write(f"Query used for retrieval: {query_for_search}")

        # Retrieving the top 5 most relevant chunks from FAISS.
        # k=5 will provide sufficient context while reducing irrelevant information.
        distances, indices = index.search(query_embedding,k=5)


        # Store the retrieved chunks in a list
        retrieved_chunks = []

        # Using FAISS retrieval results to fetch the corresponding chunk information.
        for idx in indices[0]:
            retrieved_chunks.append(chunks[idx])

        # Displaying source references and supporting passages used for answer generation.    
        st.subheader("Source References")
        
        for chunk in retrieved_chunks:
            with st.expander(
                f"{chunk['file_name']} | "
                f"Page {chunk['page_number']} | "
                f"Chunk {chunk['chunk_id']}"
            ):
                st.write(chunk["text"])


        # Combining the retrieved chunks into a single context, so that the LLM can generate a complete answer.
        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )
            
        # Displaying the context length for verification before sending it to LLM.
        st.write(f"Context length: {len(context.split())} words")

        # Creating a prompt that instructs Gemini to answer only from the retrieved document context 
        # and avoid generating unsupported information using its own training knowledge.
        prompt = f"""
        You are a document question answering assistant.

        Use only the information provided in the context below to answer the user's question.

        Do not use outside knowledge or make assumptions.
            
        If the answer cannot be determined from the provided context, respond exactly with:

        "I could not find sufficient information in the provided documents to answer this question."
            
        Context:
        {context}
            
        Question:
        {user_query}
            
        Answer:
        """

        # Generating the answer using Gemini model.
        response = llm_model.generate_content(prompt)
        # Extracting the generated answer from the response.
        final_answer = response.text

        # Displaying the user's question in chat format.
        with st.chat_message("user"):
            st.write(user_query)

        # Displaying the generated answer in chat format.    
        with st.chat_message("assistant"):
            st.write(final_answer)

        # Storing the current conversation in session memory for handling future follow-up questions.
        st.session_state.chat_history.append({
            "question": user_query,
            "answer": final_answer
        })
        
