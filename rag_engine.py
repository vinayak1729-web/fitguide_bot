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

class RAGEngine:
    def __init__(self):
        load_dotenv()
        self.FAISS_INDEX_PATH = "faiss_car_speakers_index"
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.rag_chain = None
        self.initialize()
    
    def initialize(self):
        """Initialize or load the FAISS index and RAG chain"""
        try:
            if os.path.exists(self.FAISS_INDEX_PATH):
                print("📂 Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    self.FAISS_INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing index with {self.vector_store.index.ntotal} vectors")
            else:
                print("📥 Loading CSV file...")
                loader = CSVLoader('data/tbl_Speakers_Complete.csv')
                documents = loader.load()
                print(f"✅ Loaded {len(documents)} rows from tbl_Speakers_Complete.csv")
                
                print("✂️ Splitting documents into chunks...")
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                print(f"✅ Created {len(chunks)} chunks")
                
                print("🔨 Creating FAISS index with embeddings...")
                self.vector_store = None
                batch_size = 50
                
                with tqdm(total=len(chunks), desc="🚀 Embedding documents", unit="chunk") as pbar:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        if self.vector_store is None:
                            self.vector_store = FAISS.from_documents(batch, self.embeddings)
                        else:
                            self.vector_store.add_documents(batch)
                        pbar.update(len(batch))
                
                print(f"\n✅ FAISS index created with {self.vector_store.index.ntotal} vectors")
                self.vector_store.save_local(self.FAISS_INDEX_PATH)
                print("✅ Index saved successfully!")
            
            # Initialize LLM and RAG chain
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
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
            
            self.rag_chain = (
                {"context": self.vector_store.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            print("✅ RAG Engine initialized successfully!\n")
            
        except Exception as e:
            print(f"❌ ERROR initializing RAG engine: {str(e)}")
            raise
    
    def get_answer(self, question):
        """Get answer from RAG system"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
 