"""
Document processing service for Indonesian Legal Documents.
"""

import asyncio
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from urllib.parse import urljoin, urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings, LEGAL_DOCUMENT_SOURCES
from models import DocumentMetadata, DocumentType, ChunkMetadata


class DocumentProcessor:
    """Handles document extraction, processing, and chunking."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    async def extract_content_from_url(self, url: str) -> Tuple[str, DocumentType]:
        """Extract content from a URL (HTML or PDF)."""
        try:
            response = await self._fetch_url_async(url)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type:
                return await self._extract_pdf_content(response.content), DocumentType.PDF
            else:
                return await self._extract_html_content(response.content, url), DocumentType.HTML
        
        except Exception as e:
            raise ValueError(f"Failed to extract content from {url}: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch_url_async(self, url: str) -> requests.Response:
        """Fetch URL with retry logic."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use asyncio to run requests in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.get(url, headers=headers, timeout=30)
        )
        response.raise_for_status()
        return response
    
    async def _extract_html_content(self, html_content: bytes, base_url: str) -> str:
        """Extract text content from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text from common content areas
        content_selectors = [
            '.content', '.article-content', '.document-content',
            '#content', '#main-content', '.main-content',
            'article', 'main', '.post-content'
        ]
        
        extracted_text = ""
        
        # Try specific content selectors first
        for selector in content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                extracted_text = content_area.get_text()
                break
        
        # Fallback to body content
        if not extracted_text.strip():
            body = soup.find('body')
            if body:
                extracted_text = body.get_text()
        
        # Final fallback to all text
        if not extracted_text.strip():
            extracted_text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in extracted_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    async def _extract_pdf_content(self, pdf_content: bytes) -> str:
        """Extract text content from PDF."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
            
            return '\n\n'.join(text_content)
        
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {str(e)}")
    
    def create_document_metadata(
        self,
        title: str,
        content: str,
        doc_type: DocumentType,
        source_url: Optional[str] = None,
        regulation_number: Optional[str] = None,
        year: Optional[int] = None,
        category: Optional[str] = None
    ) -> DocumentMetadata:
        """Create document metadata."""
        
        # Generate unique document ID
        doc_id = self._generate_document_id(title, content)
        
        # Auto-extract regulation info if not provided
        if not regulation_number and title:
            regulation_number = self._extract_regulation_number(title)
        
        if not year and regulation_number:
            year = self._extract_year_from_regulation(regulation_number)
        
        return DocumentMetadata(
            document_id=doc_id,
            title=title,
            source_url=source_url,
            document_type=doc_type,
            regulation_number=regulation_number,
            year=year,
            category=category or "Peraturan Perundang-undangan",
            file_size=len(content.encode('utf-8'))
        )
    
    async def chunk_document(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[Document]:
        """Split document into chunks with metadata."""
        
        # Create chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Calculate positions
            start_pos = content.find(chunk[:50]) if len(chunk) >= 50 else content.find(chunk)
            end_pos = start_pos + len(chunk) if start_pos != -1 else len(chunk)
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{metadata.document_id}_chunk_{i}",
                document_id=metadata.document_id,
                chunk_index=i,
                start_pos=max(0, start_pos),
                end_pos=end_pos,
                chunk_size=len(chunk)
            )
            
            # Create document with enhanced metadata
            doc_metadata = {
                "document_id": metadata.document_id,
                "title": metadata.title,
                "source_url": metadata.source_url,
                "document_type": metadata.document_type.value,
                "regulation_number": metadata.regulation_number,
                "year": metadata.year,
                "category": metadata.category,
                "chunk_id": chunk_metadata.chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "language": "id"
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    async def process_predefined_documents(self) -> List[Document]:
        """Process all predefined legal documents."""
        all_documents = []
        
        for doc_key, doc_info in LEGAL_DOCUMENT_SOURCES.items():
            try:
                print(f"Processing {doc_info['name']}...")
                
                # Extract content
                content, doc_type = await self.extract_content_from_url(doc_info['url'])
                
                # Create metadata
                metadata = self.create_document_metadata(
                    title=doc_info['name'],
                    content=content,
                    doc_type=doc_type,
                    source_url=doc_info['url'],
                    category="Peraturan Perundang-undangan Indonesia"
                )
                
                # Chunk document
                chunks = await self.chunk_document(content, metadata)
                all_documents.extend(chunks)
                
                print(f"✓ Processed {doc_info['name']}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"✗ Failed to process {doc_info['name']}: {str(e)}")
                continue
        
        return all_documents
    
    def _generate_document_id(self, title: str, content: str) -> str:
        """Generate unique document ID."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()[:8]
        return f"doc_{title_hash}_{content_hash}"
    
    def _extract_regulation_number(self, title: str) -> Optional[str]:
        """Extract regulation number from title."""
        import re
        
        patterns = [
            r'UU\s+No\.?\s*(\d+)\s+Tahun\s+(\d{4})',
            r'Undang-Undang\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
            r'UUD\s+(\d{4})',
            r'PP\s+No\.?\s*(\d+)\s+Tahun\s+(\d{4})',
            r'Perpres\s+No\.?\s*(\d+)\s+Tahun\s+(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                if 'UUD' in pattern:
                    return f"UUD {match.group(1)}"
                else:
                    return f"{match.group(1)}/{match.group(2)}"
        
        return None
    
    def _extract_year_from_regulation(self, regulation_number: str) -> Optional[int]:
        """Extract year from regulation number."""
        import re
        
        match = re.search(r'(\d{4})', regulation_number)
        if match:
            return int(match.group(1))
        
        return None