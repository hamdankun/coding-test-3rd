"""
Document processing service using pdfplumber

Handles PDF parsing, table extraction, text chunking, and embedding generation.
"""
from typing import Dict, List, Any, Optional
import pdfplumber
import logging
from app.core.config import settings
from app.services.table_parser import TableParser
from app.services.vector_store import VectorStore
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process PDF documents and extract structured data"""
    
    def __init__(self, db: Session = None):
        self.table_parser = TableParser()
        self.db = db
        self.vector_store = VectorStore(db=db) if db else None
    
    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document
        
        Args:
            file_path: Path to the PDF file
            document_id: Database document ID
            fund_id: Fund ID
            
        Returns:
            Processing result with statistics
        """
        # Initialize or use existing db session
        if not self.db:
            db = SessionLocal()
            should_close_db = True
        else:
            db = self.db
            should_close_db = False
        
        # Initialize vector store if not already done
        if not self.vector_store:
            self.vector_store = VectorStore(db=db)
        
        try:
            logger.info(f"Starting document processing for {file_path}")
            
            # Extract content from PDF
            pdf_content = await self._extract_pdf_content(file_path)
            
            if not pdf_content:
                return {
                    "status": "failed",
                    "error": "Could not extract content from PDF",
                    "statistics": {}
                }
            
            # Process tables
            table_results = await self._process_tables(
                pdf_content["tables"],
                document_id,
                fund_id,
                db
            )
            
            # Process text for vector storage
            text_results = await self._process_text(
                pdf_content["text"],
                document_id,
                fund_id
            )
            
            logger.info(f"Document {document_id} processed successfully")
            
            return {
                "status": "completed",
                "statistics": {
                    "tables_found": len(pdf_content["tables"]),
                    "tables_processed": table_results["count"],
                    "transactions_extracted": table_results["transactions"],
                    "text_chunks_created": text_results["chunks"],
                    "embeddings_stored": text_results["embeddings"]
                }
            }
        
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "statistics": {}
            }
        
        finally:
            if should_close_db:
                db.close()
    
    async def _extract_pdf_content(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract text and tables from PDF"""
        try:
            with pdfplumber.open(file_path) as pdf:
                tables = []
                text_content = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            tables.append({
                                "page": page_num,
                                "data": table
                            })
                    
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append({
                            "page": page_num,
                            "text": page_text
                        })
                
                return {
                    "tables": tables,
                    "text": text_content,
                    "page_count": len(pdf.pages)
                }
        
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return None
    
    async def _process_tables(
        self,
        tables: List[Dict[str, Any]],
        document_id: int,
        fund_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """Process extracted tables and store transactions"""
        processed_count = 0
        total_transactions = 0
        
        for table_info in tables:
            try:
                # Parse table
                parsed_table = self.table_parser.parse_table(table_info["data"])
                
                if "error" in parsed_table:
                    logger.warning(f"Could not parse table on page {table_info['page']}")
                    continue
                
                # Classify table
                table_type = self.table_parser.classify_table(parsed_table)
                
                if table_type == "unknown":
                    logger.debug(f"Unknown table type on page {table_info['page']}")
                    continue
                
                # Extract transactions
                transactions = self.table_parser.extract_transactions(parsed_table, table_type)
                
                if not transactions:
                    continue
                
                # Store transactions in database
                stored_count = await self._store_transactions(
                    transactions,
                    table_type,
                    fund_id,
                    document_id,
                    db
                )
                
                processed_count += 1
                total_transactions += stored_count
                
                logger.info(
                    f"Processed table {table_type}: {stored_count} transactions "
                    f"(page {table_info['page']})"
                )
            
            except Exception as e:
                logger.error(f"Error processing table on page {table_info['page']}: {e}")
                continue
        
        return {
            "count": processed_count,
            "transactions": total_transactions
        }
    
    async def _store_transactions(
        self,
        transactions: List[Dict[str, Any]],
        table_type: str,
        fund_id: int,
        document_id: int,
        db: Session
    ) -> int:
        """Store extracted transactions in database"""
        stored_count = 0
        
        try:
            for transaction in transactions:
                try:
                    if table_type == "capital_calls":
                        obj = CapitalCall(
                            fund_id=fund_id,
                            call_date=transaction.get("call_date"),
                            call_type=transaction.get("call_type", "Investment"),
                            amount=transaction.get("amount"),
                            description=transaction.get("description")
                        )
                    
                    elif table_type == "distributions":
                        obj = Distribution(
                            fund_id=fund_id,
                            distribution_date=transaction.get("distribution_date"),
                            distribution_type=transaction.get("distribution_type", "Return"),
                            is_recallable=transaction.get("is_recallable", False),
                            amount=transaction.get("amount"),
                            description=transaction.get("description")
                        )
                    
                    elif table_type == "adjustments":
                        obj = Adjustment(
                            fund_id=fund_id,
                            adjustment_date=transaction.get("adjustment_date"),
                            adjustment_type=transaction.get("adjustment_type", "Rebalance"),
                            category=transaction.get("category", "Other"),
                            amount=transaction.get("amount"),
                            is_contribution_adjustment=transaction.get("is_contribution_adjustment", False),
                            description=transaction.get("description")
                        )
                    
                    else:
                        continue
                    
                    db.add(obj)
                    stored_count += 1
                
                except Exception as e:
                    logger.error(f"Error storing transaction: {e}")
                    continue
            
            db.commit()
            logger.info(f"Stored {stored_count} transactions for fund {fund_id}")
        
        except Exception as e:
            logger.error(f"Error committing transactions: {e}")
            db.rollback()
        
        return stored_count
    
    async def _process_text(
        self,
        text_content: List[Dict[str, Any]],
        document_id: int,
        fund_id: int
    ) -> Dict[str, Any]:
        """Process text content and create embeddings"""
        chunks_created = 0
        embeddings_stored = 0
        
        try:
            # Combine all text
            full_text = " ".join([item["text"] for item in text_content if item.get("text")])
            
            if not full_text:
                return {
                    "chunks": 0,
                    "embeddings": 0
                }
            
            # Chunk text
            chunks = self._chunk_text(full_text, document_id, fund_id)
            chunks_created = len(chunks)
            
            # Store embeddings (synchronously)
            for chunk in chunks:
                try:
                    # Directly call vector store add_document synchronously
                    # Generate embedding
                    embedding = self.vector_store._get_embedding_sync(chunk["content"])
                    embedding_list = embedding.tolist()
                    
                    # Format embedding as PostgreSQL array string
                    embedding_str = f"[{','.join(str(x) for x in embedding_list)}]"
                    
                    # Get raw connection from SQLAlchemy
                    raw_conn = self.vector_store.db.connection().connection
                    cursor = raw_conn.cursor()
                    
                    # Insert using psycopg2 %s parameter style
                    import json
                    metadata_json = json.dumps({
                        "document_id": document_id,
                        "fund_id": fund_id,
                        "page": chunk.get("page"),
                        "chunk_index": chunk.get("chunk_index")
                    })
                    
                    insert_sql = """
                        INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                    """
                    
                    cursor.execute(insert_sql, [
                        document_id,
                        fund_id,
                        chunk["content"],
                        embedding_str,
                        metadata_json
                    ])
                    raw_conn.commit()
                    cursor.close()
                    embeddings_stored += 1
                
                except Exception as e:
                    logger.error(f"Error storing embedding: {e}")
                    continue
            
            logger.info(f"Created {chunks_created} text chunks and stored {embeddings_stored} embeddings")
        
        except Exception as e:
            logger.error(f"Error processing text: {e}")
        
        return {
            "chunks": chunks_created,
            "embeddings": embeddings_stored
        }
    
    def _chunk_text(
        self,
        text: str,
        document_id: int,
        fund_id: int,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text content for vector storage
        
        Args:
            text: Text to chunk
            document_id: Document ID for metadata
            fund_id: Fund ID for metadata
            chunk_size: Size of each chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            
        Returns:
            List of text chunks with metadata
        """
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        
        # Split by sentences first to avoid breaking them
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "document_id": document_id,
                        "fund_id": fund_id,
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = sentence
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "document_id": document_id,
                "fund_id": fund_id,
                "chunk_index": chunk_index
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Split on period, question mark, exclamation mark followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
