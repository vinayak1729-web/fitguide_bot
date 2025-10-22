import os
import pandas as pd
import re
from rapidfuzz import fuzz, process
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
        self.CSV_PATH = 'data/tbl_Speakers_Complete.csv'
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.rag_chain = None
        self.df = None
        self.initialize()
    
    def load_csv_data(self):
        """Load CSV data for validation"""
        try:
            self.df = pd.read_csv(self.CSV_PATH)
            print(f"✅ Loaded CSV with {len(self.df)} rows")
            
            self.df.columns = self.df.columns.str.strip()
            
            makes = self.df['Make'].dropna().astype(str).str.upper().unique().tolist()
            self.available_makes = sorted([m for m in makes if m and m.strip() and m != 'NAN'])
            print(f"✅ Found {len(self.available_makes)} unique car makes")
            
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            raise
    
    def initialize(self):
        """Initialize or load the FAISS index and RAG chain"""
        try:
            self.load_csv_data()
            
            if os.path.exists(self.FAISS_INDEX_PATH):
                print("📂 Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    self.FAISS_INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing index with {self.vector_store.index.ntotal} vectors")
            else:
                print("📥 Loading CSV file for embedding...")
                loader = CSVLoader(self.CSV_PATH)
                documents = loader.load()
                print(f"✅ Loaded {len(documents)} rows")
                
                print("✂️ Splitting documents...")
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                print(f"✅ Created {len(chunks)} chunks")
                
                print("🔨 Creating FAISS index...")
                self.vector_store = None
                batch_size = 50
                
                with tqdm(total=len(chunks), desc="🚀 Embedding", unit="chunk") as pbar:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        if self.vector_store is None:
                            self.vector_store = FAISS.from_documents(batch, self.embeddings)
                        else:
                            self.vector_store.add_documents(batch)
                        pbar.update(len(batch))
                
                print(f"\n✅ FAISS index created")
                self.vector_store.save_local(self.FAISS_INDEX_PATH)
                print("✅ Index saved!")
            
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            template = """You are a CAR AUDIO EXPERT. Provide speaker details clearly.

Make: {make}
Model: {model}
Year: {year}
Component: {component}

Context: {context}

Provide clear speaker details including sizes, locations, and installation notes."""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            self.rag_chain = (
                {
                    "context": self.vector_store.as_retriever(search_kwargs={"k": 5}), 
                    "make": lambda x: x.get("make", ""),
                    "model": lambda x: x.get("model", ""),
                    "year": lambda x: x.get("year", ""),
                    "component": lambda x: x.get("component", ""),
                    "question": lambda x: x.get("question", "")
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            print("✅ RAG Engine initialized!\n")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            raise
    
    def fuzzy_match_make(self, user_input, threshold=70):
        """Fuzzy match car make"""
        user_input_upper = user_input.upper().strip()
        
        if user_input_upper in self.available_makes:
            return True, user_input_upper, []
        
        matches = process.extract(
            user_input_upper, 
            self.available_makes, 
            scorer=fuzz.ratio,
            limit=5
        )
        
        if matches and matches[0][1] >= threshold:
            best_match = matches[0][0]
            suggestions = [m[0] for m in matches if m[1] >= 60][:5]
            return True, best_match, suggestions
        
        suggestions = [m[0] for m in matches if m[1] >= 50][:5]
        return False, None, suggestions
    
    def fuzzy_match_model(self, user_input, available_models, threshold=65):
        """Fuzzy match model - LOWERED threshold for better matching"""
        if not available_models:
            return None, []
        
        user_input_clean = user_input.upper().strip()
        
        # Exact match
        for model in available_models:
            model_clean = model.upper().strip()
            if user_input_clean == model_clean:
                return model, []
            # Check model without year
            if '(' in model:
                model_without_year = model.split('(')[0].strip().upper()
                if user_input_clean == model_without_year:
                    return model, []
        
        # Fuzzy match
        matches = process.extract(
            user_input,
            available_models,
            scorer=fuzz.token_set_ratio,
            limit=8
        )
        
        if matches and matches[0][1] >= threshold:
            best_match = matches[0][0]
            suggestions = [m[0] for m in matches if m[1] >= 50][:8]
            return best_match, suggestions
        
        suggestions = [m[0] for m in matches if m[1] >= 40][:8]
        return None, suggestions
    
    def search_model_in_make(self, make, model_query):
        """Search for model within a specific make"""
        models = self.get_models_and_years_for_make(make)
        
        # Try fuzzy match
        matched_model, suggestions = self.fuzzy_match_model(model_query, models, threshold=60)
        
        return matched_model, suggestions
    
    def extract_car_info_from_text(self, text):
        """Extract both make and model from text like 'GX 550 Lexus'"""
        text_upper = text.upper()
        
        # Check for make in text
        found_make = None
        for make in self.available_makes:
            if make in text_upper:
                found_make = make
                # Remove make from text to extract model
                text_upper = text_upper.replace(make, '').strip()
                break
        
        # If make found, try to match remaining text as model
        if found_make:
            models = self.get_models_and_years_for_make(found_make)
            matched_model, _ = self.fuzzy_match_model(text_upper, models, threshold=50)
            return found_make, matched_model
        
        return None, None
    
    def validate_make(self, make):
        """Check if make exists"""
        make_upper = make.upper().strip()
        if make_upper in self.available_makes:
            return True, make_upper
        
        close_matches = [m for m in self.available_makes if make_upper in m or m in make_upper]
        if close_matches:
            return True, close_matches[0]
        
        return False, None
    
    def get_models_and_years_for_make(self, make):
        """Get models with years"""
        try:
            filtered_df = self.df[self.df['Make'].str.upper() == make.upper()]
            
            model_year_set = set()
            for _, row in filtered_df.iterrows():
                model = str(row.get('Model', ''))
                year = str(row.get('Year', ''))
                
                if model and model.strip() and model.upper() != 'NAN':
                    if year and year.strip() and year.upper() != 'NAN' and year != 'nan':
                        year_clean = year.split('.')[0]
                        model_year = f"{model} ({year_clean})"
                    else:
                        model_year = model
                    
                    model_year_set.add(model_year)
            
            return sorted(list(model_year_set))
        except Exception as e:
            print(f"Error: {str(e)}")
            return []
    
    def parse_model_year(self, model_year_input):
        """Parse model and year"""
        if '(' in model_year_input and ')' in model_year_input:
            parts = model_year_input.split('(')
            model = parts[0].strip()
            year = parts[1].replace(')', '').strip()
            return model, year
        else:
            return model_year_input.strip(), None
    
    def get_components_for_vehicle(self, make, model, year=None):
        """Get components"""
        try:
            if year:
                filtered = self.df[
                    (self.df['Make'].str.upper() == make.upper()) & 
                    (self.df['Model'].astype(str).str.upper() == model.upper()) &
                    (self.df['Year'].astype(str).str.contains(year, na=False))
                ]
            else:
                filtered = self.df[
                    (self.df['Make'].str.upper() == make.upper()) & 
                    (self.df['Model'].astype(str).str.upper() == model.upper())
                ]
            
            if not filtered.empty:
                components = filtered['searchSectionName'].dropna().astype(str).unique().tolist()
                components = [c for c in components if c and c.strip() and c.upper() != 'NAN']
                return sorted(components)
            return []
        except Exception as e:
            print(f"Error: {str(e)}")
            return []
    
    def is_valid_value(self, value):
        """Check if value is valid"""
        if pd.isna(value):
            return False
        str_value = str(value).strip()
        if not str_value or str_value.upper() in ['NAN', 'NONE', '']:
            return False
        return True
    
    def get_speaker_data_table(self, make, model, component, year=None):
        """Get speaker data"""
        try:
            if year:
                filtered = self.df[
                    (self.df['Make'].str.upper() == make.upper()) & 
                    (self.df['Model'].astype(str).str.upper() == model.upper()) &
                    (self.df['searchSectionName'].astype(str).str.upper() == component.upper()) &
                    (self.df['Year'].astype(str).str.contains(year, na=False))
                ]
            else:
                filtered = self.df[
                    (self.df['Make'].str.upper() == make.upper()) & 
                    (self.df['Model'].astype(str).str.upper() == model.upper()) &
                    (self.df['searchSectionName'].astype(str).str.upper() == component.upper())
                ]
            
            if not filtered.empty:
                table_data = []
                seen_entries = set()
                
                row = filtered.iloc[0]
                
                for i in range(1, 4):
                    size_col = f'{i}SpeakerSize'
                    location_col = f'{i}SpeakerLocation'
                    note_col = f'{i}DisplayNote'
                    
                    size = row.get(size_col)
                    location = row.get(location_col)
                    note = row.get(note_col)
                    
                    if self.is_valid_value(size):
                        size_str = str(size).strip()
                        location_str = str(location).strip() if self.is_valid_value(location) else 'N/A'
                        note_str = str(note).strip() if self.is_valid_value(note) else 'N/A'
                        
                        entry_key = f"{size_str}|{location_str}"
                        
                        if entry_key not in seen_entries:
                            seen_entries.add(entry_key)
                            table_data.append({
                                'Size': size_str,
                                'Location': location_str,
                                'Notes': note_str
                            })
                
                return table_data
            return []
        except Exception as e:
            print(f"Error: {str(e)}")
            return []
    
    def get_answer(self, user_message, current_stage, user_data):
        """Get answer with improved matching"""
        try:
            user_input = user_message.strip()
            user_input_lower = user_input.lower()
            
            # Check for back command
            if user_input_lower in ['back', 'go back', '← back']:
                if current_stage == 'ask_model':
                    return (
                        "👈 Going back...\n\n**Select car brand:**",
                        {'current_stage': 'ask_make', 'user_data': {}, 'buttons': self.available_makes[:12]}
                    )
                elif current_stage == 'ask_component':
                    make = user_data.get('make', '')
                    models = self.get_models_and_years_for_make(make)
                    return (
                        f"👈 Going back...\n\n**Choose {make} model:**",
                        {'current_stage': 'ask_model', 'user_data': {'make': make}, 'buttons': models[:12]}
                    )
            
            # Natural language with make AND model
            if any(kw in user_input_lower for kw in ['show', 'tell', 'find', 'need', 'speaker']):
                found_make, found_model = self.extract_car_info_from_text(user_input)
                
                if found_make:
                    user_data['make'] = found_make
                    
                    if found_model:
                        # Both make and model found
                        model, year = self.parse_model_year(found_model)
                        user_data['model'] = model
                        user_data['year'] = year
                        
                        components = self.get_components_for_vehicle(found_make, model, year)
                        if components:
                            return (
                                f"✅ Found **{found_make} {found_model}**!\n\n🔧 **Choose component:**",
                                {'current_stage': 'ask_component', 'user_data': user_data, 'buttons': components}
                            )
                    else:
                        # Only make found
                        models_years = self.get_models_and_years_for_make(found_make)
                        if models_years:
                            return (
                                f"✅ Found **{found_make}**!\n\n📋 **Choose model:**",
                                {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': models_years[:12]}
                            )
            
            # Greeting
            if current_stage == 'greet':
                greetings = ['hi', 'hello', 'hey', 'howdy', 'greetings', 'good morning', 
                           'good afternoon', 'good evening', 'start', 'begin']
                
                if any(greeting in user_input_lower for greeting in greetings):
                    return (
                        "👋 **Welcome to Car Audio Assistant!**\n\n"
                        "🚗 **Step 1:** Select your car brand:",
                        {'current_stage': 'ask_make', 'user_data': {}, 'buttons': self.available_makes[:12]}
                    )
                else:
                    return (
                        "👋 Hi! Say **'hi'** or **'hello'** to start!",
                        {'current_stage': 'greet', 'user_data': {}, 'buttons': []}
                    )
            
            # Ask Make
            elif current_stage == 'ask_make':
                found, matched_make, suggestions = self.fuzzy_match_make(user_input)
                
                if found:
                    user_data['make'] = matched_make
                    models_years = self.get_models_and_years_for_make(matched_make)
                    
                    if models_years:
                        msg = f"✅ **{matched_make}** selected!\n\n📋 **Step 2:** Choose your model:"
                        if matched_make.upper() != user_input.upper():
                            msg = f"✅ Did you mean **{matched_make}**?\n\n📋 **Choose your model:**"
                        
                        return (msg, {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': models_years[:12]})
                else:
                    if suggestions:
                        return (
                            f"🤔 **'{user_input}'** not found.\n\n**Did you mean:**",
                            {'current_stage': 'ask_make', 'user_data': {}, 'buttons': suggestions[:6]}
                        )
                    else:
                        return (
                            f"❌ **'{user_input}'** not found.\n\n**Available brands:**",
                            {'current_stage': 'ask_make', 'user_data': {}, 'buttons': self.available_makes[:12]}
                        )
            
            # Ask Model - IMPROVED
            elif current_stage == 'ask_model':
                make = user_data.get('make', '')
                models_years = self.get_models_and_years_for_make(make)
                
                matched_model, suggestions = self.fuzzy_match_model(user_input, models_years, threshold=60)
                
                if matched_model:
                    model, year = self.parse_model_year(matched_model)
                    user_data['model'] = model
                    user_data['year'] = year
                    
                    components = self.get_components_for_vehicle(make, model, year)
                    
                    if components:
                        year_display = f" ({year})" if year else ""
                        msg = f"✅ **{make} {model}{year_display}** selected!\n\n🔧 **Step 3:** Choose component:"
                        
                        if matched_model.upper() != user_input.upper():
                            msg = f"✅ Did you mean **{model}{year_display}**?\n\n🔧 **Choose component:**"
                        
                        return (msg, {'current_stage': 'ask_component', 'user_data': user_data, 'buttons': ['← Back'] + components})
                    else:
                        return (
                            f"Sorry, no data for {make} {model}.\n\nChoose another model:",
                            {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': models_years[:12]}
                        )
                else:
                    if suggestions:
                        return (
                            f"🤔 **'{user_input}'** not found.\n\n**Did you mean:**",
                            {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': ['← Back'] + suggestions[:8]}
                        )
                    else:
                        return (
                            f"❌ **'{user_input}'** not found.\n\n**Available models:**",
                            {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': ['← Back'] + models_years[:12]}
                        )
            
            # Component
            elif current_stage == 'ask_component':
                make = user_data.get('make', '')
                model = user_data.get('model', '')
                year = user_data.get('year')
                components = self.get_components_for_vehicle(make, model, year)
                
                asking_keywords = ['what', 'which', 'available', 'have', 'show', 'list']
                if any(keyword in user_input_lower for keyword in asking_keywords):
                    return (
                        f"📋 **Available components:**\n\nClick to view:",
                        {'current_stage': 'ask_component', 'user_data': user_data, 'buttons': ['← Back'] + components}
                    )
                
                component_found = None
                for comp in components:
                    if user_input.upper() in comp.upper() or comp.upper() in user_input.upper():
                        component_found = comp
                        break
                
                if component_found:
                    table_data = self.get_speaker_data_table(make, model, component_found, year)
                    
                    return (
                        None,
                        {
                            'current_stage': 'show_results',
                            'user_data': user_data,
                            'component': component_found,
                            'table_data': table_data,
                            'buttons': ['🔄 Search Again']
                        }
                    )
                else:
                    return (
                        f"❌ Not found.\n\n**Available:**",
                        {'current_stage': 'ask_component', 'user_data': user_data, 'buttons': ['← Back'] + components}
                    )
            
            # Results
            elif current_stage == 'show_results':
                if '🔄' in user_input or 'search' in user_input_lower or 'again' in user_input_lower:
                    return (
                        "👋 **New search!**\n\n**Select brand:**",
                        {'current_stage': 'ask_make', 'user_data': {}, 'buttons': self.available_makes[:12]}
                    )
                
                found_make, found_model = self.extract_car_info_from_text(user_input)
                if found_make:
                    user_data = {'make': found_make}
                    models_years = self.get_models_and_years_for_make(found_make)
                    
                    if models_years:
                        return (
                            f"✅ Switching to **{found_make}**!\n\n**Choose model:**",
                            {'current_stage': 'ask_model', 'user_data': user_data, 'buttons': models_years[:12]}
                        )
            
            return (
                "❌ I didn't understand. Say **'hi'** to start!",
                {'current_stage': 'greet', 'user_data': {}, 'buttons': []}
            )
                
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return (
                f"Sorry, error occurred.\n\nSay **'hi'** to restart.",
                {'current_stage': 'greet', 'user_data': {}, 'buttons': []}
            )
