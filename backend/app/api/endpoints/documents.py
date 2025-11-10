"""
Document API endpoints
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from datetime import datetime
from app.db.session import get_db
from app.models.document import Document
from app.models.fund import Fund
from app.schemas.document import (
    Document as DocumentSchema,
    DocumentUploadResponse,
    DocumentStatus
)
from app.services.document_processor import DocumentProcessor
from app.core.config import settings

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fund_id: int = None,
    db: Session = Depends(get_db)
):
    """Upload and process a PDF document"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    # If fund_id not provided, create default fund or use first fund
    if not fund_id:
        existing_fund = db.query(Fund).first()
        if existing_fund:
            fund_id = existing_fund.id
        else:
            # Create a default fund if none exist
            default_fund = Fund(
                name="Default Fund",
                gp_name="Unknown GP",
                fund_type="Unknown",
                vintage_year=2024
            )
            db.add(default_fund)
            db.commit()
            db.refresh(default_fund)
            fund_id = default_fund.id
    else:
        # Verify fund exists
        fund = db.query(Fund).filter(Fund.id == fund_id).first()
        if not fund:
            raise HTTPException(
                status_code=404,
                detail=f"Fund with ID {fund_id} not found. Please create a fund first."
            )
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create document record
    document = Document(
        fund_id=fund_id,
        file_name=file.filename,
        file_path=file_path,
        parsing_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Start background processing
    background_tasks.add_task(
        process_document_task,
        document.id,
        file_path,
        fund_id
    )
    
    return DocumentUploadResponse(
        document_id=document.id,
        task_id=None,
        status="pending",
        message="Document uploaded successfully. Processing started."
    )


async def process_document_task(document_id: int, file_path: str, fund_id: int):
    """Background task to process document"""
    from app.db.session import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Update status to processing
        document = db.query(Document).filter(Document.id == document_id).first()
        document.parsing_status = "processing"
        db.commit()
        
        # Process document (pass db session to DocumentProcessor)
        processor = DocumentProcessor(db=db)
        result = await processor.process_document(file_path, document_id, fund_id)
        
        # Update status
        document.parsing_status = result["status"]
        if result["status"] == "failed":
            document.error_message = result.get("error")
        db.commit()
        
    except Exception as e:
        document = db.query(Document).filter(Document.id == document_id).first()
        document.parsing_status = "failed"
        document.error_message = str(e)
        db.commit()
    finally:
        db.close()


@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(document_id: int, db: Session = Depends(get_db)):
    """Get document parsing status"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentStatus(
        document_id=document.id,
        status=document.parsing_status,
        error_message=document.error_message
    )


@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document details"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.get("/", response_model=List[DocumentSchema])
async def list_documents(
    fund_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all documents"""
    query = db.query(Document)
    
    if fund_id:
        query = query.filter(Document.fund_id == fund_id)
    
    documents = query.offset(skip).limit(limit).all()
    return documents


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    if document.file_path and os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete database record
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}
