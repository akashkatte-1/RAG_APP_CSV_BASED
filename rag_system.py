import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, gemini_api_key: str):
        """Initialize the RAG system with Gemini API and ChromaDB"""
        self.gemini_api_key = gemini_api_key
        self.setup_gemini()
        self.setup_chromadb()
        self.setup_embeddings()
        
    def setup_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise
    
    def setup_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="csv_documents",
                metadata={"description": "CSV document chunks for RAG"}
            )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def setup_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def process_csv(self, filepath: str) -> Tuple[bool, str]:
        """Process CSV file and add to vector database"""
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Generate document chunks from CSV
            chunks = self.create_chunks_from_csv(df, filepath)
            
            if not chunks:
                return False, "No valid chunks created from CSV file"
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
            return True, f"Processed {len(chunks)} chunks from {len(df)} rows"
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return False, str(e)
    
    def create_chunks_from_csv(self, df: pd.DataFrame, filepath: str) -> List[Dict[str, Any]]:
        """Create text chunks from CSV data"""
        chunks = []
        filename = os.path.basename(filepath)
        
        try:
            # Strategy 1: Each row as a chunk
            for idx, row in df.iterrows():
                # Convert row to text representation
                row_text = self.row_to_text(row, df.columns)
                
                chunk_id = hashlib.md5(f"{filename}_{idx}_{row_text}".encode()).hexdigest()
                
                chunk = {
                    'id': chunk_id,
                    'text': row_text,
                    'metadata': {
                        'source': filename,
                        'row_index': int(idx),
                        'chunk_type': 'row',
                        'timestamp': datetime.now().isoformat()
                    }
                }
                chunks.append(chunk)
            
            # Strategy 2: Column summaries as chunks
            for col in df.columns:
                if df[col].dtype in ['object', 'string']:
                    # For text columns, create a summary chunk
                    unique_values = df[col].dropna().unique()[:20]  # Top 20 unique values
                    col_text = f"Column '{col}' contains: {', '.join(map(str, unique_values))}"
                elif df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, create statistical summary
                    stats = df[col].describe()
                    col_text = f"Column '{col}' statistics: mean={stats['mean']:.2f}, min={stats['min']}, max={stats['max']}, std={stats['std']:.2f}"
                else:
                    col_text = f"Column '{col}' contains {df[col].dtype} data with {df[col].count()} non-null values"
                
                chunk_id = hashlib.md5(f"{filename}_col_{col}_{col_text}".encode()).hexdigest()
                
                chunk = {
                    'id': chunk_id,
                    'text': col_text,
                    'metadata': {
                        'source': filename,
                        'column_name': col,
                        'chunk_type': 'column_summary',
                        'timestamp': datetime.now().isoformat()
                    }
                }
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks from CSV")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return []
    
    def row_to_text(self, row: pd.Series, columns: List[str]) -> str:
        """Convert a pandas row to readable text"""
        text_parts = []
        for col in columns:
            value = row[col]
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        
        return " | ".join(text_parts)
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question]).tolist()[0]
            
            # Search ChromaDB for relevant chunks
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'answer': "I don't have any relevant information to answer your question. Please upload some CSV files first.",
                    'sources': []
                }
            
            # Prepare context from retrieved chunks
            context_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            context = "\n\n".join([
                f"Source: {meta.get('source', 'Unknown')}\n{chunk}"
                for chunk, meta in zip(context_chunks, metadatas)
            ])
            
            # Generate answer using Gemini
            prompt = self.create_prompt(question, context)
            response = self.model.generate_content(prompt)
            
            # Prepare sources information
            sources = []
            for meta in metadatas:
                source_info = {
                    'source': meta.get('source', 'Unknown'),
                    'type': meta.get('chunk_type', 'Unknown'),
                }
                if 'row_index' in meta:
                    source_info['row_index'] = meta['row_index']
                if 'column_name' in meta:
                    source_info['column_name'] = meta['column_name']
                sources.append(source_info)
            
            return {
                'answer': response.text,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': []
            }
    
    def create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for Gemini with context"""
        prompt = f"""You are a helpful AI assistant that answers questions based on CSV data provided as context.

Context from CSV files:
{context}

Question: {question}

Instructions:
1. Answer the question based only on the provided context
2. Be specific and cite which data sources you're referencing
3. If you're making calculations or comparisons, show your reasoning
4. Keep your answer concise but comprehensive

Answer:"""
        
        return prompt
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            collection_count = self.collection.count()
            return {
                'documents_in_database': collection_count,
                'embedding_model': 'all-MiniLM-L6-v2',
                'gemini_model': 'gemini-pro'
            }
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {'error': str(e)}
    
    def reset_database(self) -> Tuple[bool, str]:
        """Reset the vector database"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection("csv_documents")
            
            # Recreate the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="csv_documents",
                metadata={"description": "CSV document chunks for RAG"}
            )
            
            logger.info("Database reset successfully")
            return True, "Database reset successfully"
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False, str(e)
