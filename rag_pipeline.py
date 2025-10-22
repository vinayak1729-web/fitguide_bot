import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

# Load environment variables
load_dotenv()
print("✅ KEY LOADED:", "YES!" if os.getenv("OPENAI_API_KEY") else "NO!")

# Check if we have a saved index
FAISS_INDEX_PATH = "faiss_car_speakers_index"

try:
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Try to load existing index
    if os.path.exists(FAISS_INDEX_PATH):
        print("📂 Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded existing index with {vector_store.index.ntotal} vectors")
    else:
        # Create new index from single CSV
        print("📥 Loading CSV file...")
        loader = CSVLoader('data/tbl_Speakers_Complete.csv')
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} rows from tbl_Speakers_Complete.csv")
        
        # Split documents into chunks
        print("✂️ Splitting documents into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Create FAISS vector store with progress bar
        print("🔨 Creating FAISS index with embeddings...")
        print("⏳ This may take a few minutes depending on data size...\n")
        
        # Initialize vector store with first document
        vector_store = None
        
        # Process documents in batches with progress bar
        batch_size = 50  # Process 50 chunks at a time to show progress
        
        with tqdm(total=len(chunks), desc="🚀 Embedding documents", unit="chunk", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                if vector_store is None:
                    # Create initial vector store with first batch
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    # Add subsequent batches to existing vector store
                    vector_store.add_documents(batch)
                
                # Update progress bar
                pbar.update(len(batch))
        
        print(f"\n✅ FAISS index created with {vector_store.index.ntotal} vectors")
        
        # Save the index for future use
        print("💾 Saving FAISS index to disk...")
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"✅ Index saved successfully!\n")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create prompt template
    template = """You are a CAR AUDIO EXPERT ASSISTANT. Answer using ONLY the speaker data provided.

Available Data:
- Speaker specifications and details
- Vehicle compatibility information
- Speaker sizes, locations, and installation notes

Context: {context}
Question: {question}

Instructions:
1. Answer whether the product the user is asking about exists in the database
2. If not found, suggest similar products based on the available data
3. Provide specific details like sizes, locations, and compatibility when available
4. Be helpful and informative

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain with increased retrieval results
    rag_chain = (
        {"context": vector_store.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Start chat interface
    print("="*70)
    print("🚗 CAR AUDIO BOT Ready! Ask about speakers and vehicles.")
    print("="*70)
    print("💡 Tips:")
    print("   - Ask about specific car models and years")
    print("   - Query speaker sizes and locations")
    print("   - Ask about speaker compatibility")
    print("   - Type 'quit' or 'exit' to end the chat")
    print("="*70 + "\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using Car Audio Bot! Goodbye!")
            break
        
        if not question:
            print("⚠️ Please enter a question.\n")
            continue
        
        try:
            print("\n🔍 Searching...\n")
            answer = rag_chain.invoke(question)
            print(f"🤖 {answer}\n")
            print("─" * 70 + "\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find CSV file!")
    print(f"   Make sure 'data/tbl_Speakers_Complete.csv' exists in the data folder")
    print(f"   Details: {str(e)}")
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
