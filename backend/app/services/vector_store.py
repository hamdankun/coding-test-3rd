"""
Vector store service using pgvector (PostgreSQL extension)

Implements vector storage and semantic search using PostgreSQL with pgvector extension.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import json
import psycopg2
from sqlalchemy.orm import Session
from sqlalchemy import text, create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """pgvector-based vector store for document embeddings"""
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
        self.embeddings = self._initialize_embeddings()
        self._ensure_extension()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            # Fallback to local embeddings
            logger.info("Using Hugging Face embeddings (local)")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _ensure_extension(self):
        """Ensure pgvector extension is enabled and tables are created"""
        try:
            # Enable pgvector extension
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")
            
            # Determine embedding dimension
            dimension = 1536 if settings.OPENAI_API_KEY else 384
            
            # Create embeddings table
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER,
                fund_id INTEGER,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (fund_id) REFERENCES funds(id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS document_embeddings_fund_idx 
            ON document_embeddings(fund_id);
            
            CREATE INDEX IF NOT EXISTS document_embeddings_document_idx 
            ON document_embeddings(document_id);
            
            CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
            ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            
            self.db.execute(text(create_table_sql))
            self.db.commit()
            logger.info("Vector store tables created/verified")
        
        except Exception as e:
            logger.error(f"Error ensuring pgvector extension: {e}")
            self.db.rollback()
            raise
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Add a document chunk to the vector store
        
        Args:
            content: Text content to embed
            metadata: Metadata including document_id, fund_id, etc.
        """
        try:
            # Generate embedding
            embedding = self._get_embedding_sync(content)
            embedding_list = embedding.tolist()
            
            # Convert metadata dict to JSON string
            metadata_json = json.dumps(metadata)
            
            # Format embedding as PostgreSQL array string: [0.1, 0.2, ...]
            embedding_str = f"[{','.join(str(x) for x in embedding_list)}]"
            
            # Get raw connection from SQLAlchemy
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()
            
            # Insert using psycopg2 %s parameter style
            insert_sql = """
                INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb)
            """
            
            cursor.execute(insert_sql, [
                metadata.get('document_id'),
                metadata.get('fund_id'),
                content,
                embedding_str,
                metadata_json
            ])
            raw_conn.commit()
            cursor.close()
            logger.debug(f"Added embedding for document {metadata.get('document_id')}")
        
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"fund_id": 1})
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Generate query embedding (synchronously for now)
            query_embedding = self._get_embedding_sync(query)
            embedding_list = query_embedding.tolist()
            
            # Format embedding as PostgreSQL array string: [0.1, 0.2, ...]
            embedding_str = f"[{','.join(str(x) for x in embedding_list)}]"
            
            # Build query with optional filters
            where_clause = ""
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    if key in ["document_id", "fund_id"]:
                        conditions.append(f"{key} = {value}")
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get raw connection from SQLAlchemy
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()
            
            # Search using cosine distance (<=> operator) with psycopg2 %s style
            search_sql = f"""
                SELECT 
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM document_embeddings
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            cursor.execute(search_sql, [embedding_str, embedding_str, k])
            
            # Format results
            results = []
            for row in cursor.fetchall():
                similarity = float(row[5]) if row[5] else 0.0
                
                # Only include results above similarity threshold
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    results.append({
                        "id": row[0],
                        "document_id": row[1],
                        "fund_id": row[2],
                        "content": row[3],
                        "metadata": row[4],
                        "score": similarity
                    })
            
            cursor.close()
            logger.debug(f"Found {len(results)} relevant documents for query")
            return results
        
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (async wrapper)"""
        try:
            return self._get_embedding_sync(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _get_embedding_sync(self, text: str) -> np.ndarray:
        """Generate embedding for text (synchronous)"""
        try:
            if hasattr(self.embeddings, 'embed_query'):
                # LangChain embeddings
                embedding = self.embeddings.embed_query(text)
            else:
                # Hugging Face embeddings
                embedding = self.embeddings.encode(text)
            
            return np.array(embedding, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def clear(self, fund_id: Optional[int] = None):
        """
        Clear the vector store
        
        Args:
            fund_id: If provided, only clear embeddings for this fund
        """
        try:
            if fund_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                self.db.execute(delete_sql, {"fund_id": fund_id})
                logger.info(f"Cleared embeddings for fund {fund_id}")
            else:
                delete_sql = text("DELETE FROM document_embeddings")
                self.db.execute(delete_sql)
                logger.info("Cleared all embeddings")
            
            self.db.commit()
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            self.db.rollback()
    
    def get_document_count(self, fund_id: Optional[int] = None) -> int:
        """Get number of stored embeddings"""
        try:
            if fund_id:
                sql = text("SELECT COUNT(*) FROM document_embeddings WHERE fund_id = :fund_id")
                result = self.db.execute(sql, {"fund_id": fund_id}).scalar()
            else:
                sql = text("SELECT COUNT(*) FROM document_embeddings")
                result = self.db.execute(sql).scalar()
            
            return result or 0
        
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
