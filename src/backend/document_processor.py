"""
Enhanced Document Processor for Indonesian Legal RAG Assistant with OCR and complete text processing pipeline.
"""

import asyncio
import os
import re
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

# PDF and OCR processing
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings

# FastAPI and async
from concurrent.futures import ThreadPoolExecutor
import aiofiles

from models import DocumentMetadata, DocumentType
from config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor with OCR and complete text processing pipeline."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.setup_nltk()
        self.setup_indonesian_nlp()
        self.setup_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # OCR configuration
        self.tesseract_config = r'--oem 3 --psm 6 -l ind+eng'
        
    def setup_nltk(self):
        """Setup NLTK downloads."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stemmer = PorterStemmer()
    
    def setup_indonesian_nlp(self):
        """Setup Indonesian NLP tools."""
        # Indonesian stemmer
        factory = StemmerFactory()
        self.indonesian_stemmer = factory.create_stemmer()
        
        # Indonesian stopwords
        stopword_factory = StopWordRemoverFactory()
        self.indonesian_stopword_remover = stopword_factory.create_stop_word_remover()
        
        # Combined stopwords (Indonesian + English)
        indonesian_stopwords = set(stopword_factory.get_stop_words())
        english_stopwords = set(stopwords.words('english'))
        self.combined_stopwords = indonesian_stopwords.union(english_stopwords)
    
    def setup_embeddings(self):
        """Setup Ollama embeddings."""
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",  
                base_url="http://localhost:11434"  
            )
            logger.info("✅ Ollama embeddings initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama embeddings: {e}")
            raise
    
    async def load_documents_from_folder(self, folder_path: str = "documents") -> List[Document]:
        """Load and process all PDF documents from a folder."""
        documents = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.warning(f"Documents folder not found: {folder_path}")
            return documents
        
        pdf_files = list(folder_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_file.name}")
                
                # Read PDF file
                async with aiofiles.open(pdf_file, 'rb') as f:
                    pdf_content = await f.read()
                
                # Extract text with OCR
                extracted_text = await self._extract_pdf_with_ocr(pdf_content)
                
                if not extracted_text or len(extracted_text.strip()) < 100:
                    logger.warning(f"Skipping {pdf_file.name}: insufficient content")
                    continue
                
                # Create metadata
                metadata = self.create_document_metadata(
                    title=pdf_file.stem,
                    content=extracted_text,
                    doc_type=DocumentType.PDF,
                    source_url=str(pdf_file),
                    file_path=str(pdf_file)
                )
                
                # Process document through complete pipeline
                doc_chunks = await self.process_document_pipeline(extracted_text, metadata)
                documents.extend(doc_chunks)
                
                logger.info(f"✅ Processed {pdf_file.name}: {len(doc_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✅ Total documents processed: {len(documents)} chunks from {len(pdf_files)} PDFs")
        return documents
    
    async def _extract_pdf_with_ocr(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF and OCR fallback."""
        try:
            # First try PyMuPDF for text extraction
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text_content = []
            
            for page_num, page in enumerate(doc):
                # Try direct text extraction first
                text = page.get_text()
                
                if text.strip():
                    text_content.append(text)
                else:
                    # Fallback to OCR for image-based PDFs
                    logger.info(f"Using OCR for page {page_num + 1}")
                    ocr_text = await self._ocr_page(page)
                    if ocr_text:
                        text_content.append(ocr_text)
            
            doc.close()
            full_text = "\n\n".join(text_content)
            
            # Clean extracted text
            return self._clean_extracted_text(full_text)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    async def _ocr_page(self, page) -> str:
        """Perform OCR on a PDF page."""
        try:
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to OpenCV format for preprocessing
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(opencv_image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed for page: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply deskewing
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only rotate if angle is significant
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from PDFs."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\nHalaman \d+.*?\n', '\n', text)
        text = re.sub(r'\nPage \d+.*?\n', '\n', text)
        
        # Fix common OCR errors for Indonesian text
        replacements = {
            'Kepres': 'Keputusan Presiden',
            'Perpres': 'Peraturan Presiden', 
            'PP': 'Peraturan Pemerintah',
            'UU': 'Undang-Undang',
            'Permen': 'Peraturan Menteri'
        }
        
        for old, new in replacements.items():
            text = re.sub(f'\\b{old}\\b', new, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    async def process_document_pipeline(self, content: str, metadata: DocumentMetadata) -> List[Document]:
        """Complete document processing pipeline."""
        try:
            # 1. Text cleaning and normalization
            cleaned_text = self._normalize_text(content)
            
            # 2. Tokenization
            tokens = self._tokenize_text(cleaned_text)
            
            # 3. Stopword removal
            filtered_tokens = self._remove_stopwords(tokens)
            
            # 4. Stemming
            stemmed_tokens = self._stem_tokens(filtered_tokens)
            
            # 5. Reconstruct text for chunking
            processed_text = ' '.join(stemmed_tokens)
            
            # 6. Chunking
            chunks = await self.chunk_document(processed_text, metadata)
            
            # 7. Generate embeddings for each chunk
            for chunk in chunks:
                try:
                    # Generate embedding
                    embedding = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.embeddings.embed_query,
                        chunk.page_content
                    )
                    chunk.metadata['embedding'] = embedding
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk: {e}")
                    # Continue without embedding
            
            return chunks
            
        except Exception as e:
            logger.error(f"Document processing pipeline failed: {e}")
            return []
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep Indonesian characters
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK."""
        try:
            tokens = word_tokenize(text, language='indonesian')
            # Filter out single characters and purely numeric tokens
            tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return text.split()
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.combined_stopwords]
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens using Indonesian stemmer."""
        try:
            # Join tokens to use Sastrawi stemmer
            text = ' '.join(tokens)
            stemmed_text = self.indonesian_stemmer.stem(text)
            return stemmed_text.split()
        except Exception as e:
            logger.error(f"Stemming failed: {e}")
            # Fallback to English stemmer
            return [self.stemmer.stem(token) for token in tokens]
    
    async def chunk_document(self, content: str, metadata: DocumentMetadata) -> List[Document]:
        """Split document into chunks."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                # Create chunk metadata
                chunk_metadata = {
                    "document_id": metadata.document_id,
                    "title": metadata.title,
                    "document_type": metadata.document_type.value,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_url": metadata.source_url,
                    "regulation_number": metadata.regulation_number,
                    "year": metadata.year,
                    "category": metadata.category,
                    "created_at": metadata.created_at.isoformat(),
                    "file_path": metadata.metadata.get("file_path")
                }
                
                # Create Document object
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            return []
    
    def create_document_metadata(
        self, 
        title: str, 
        content: str, 
        doc_type: DocumentType,
        source_url: Optional[str] = None,
        regulation_number: Optional[str] = None,
        year: Optional[int] = None,
        category: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> DocumentMetadata:
        """Create document metadata."""
        # Generate document ID from content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        document_id = f"doc_{content_hash[:12]}"
        
        # Auto-detect category from title if not provided
        if not category:
            category = self._detect_category(title, content)
        
        # Auto-detect year from content if not provided
        if not year:
            year = self._extract_year(content)
        
        # Auto-detect regulation number if not provided
        if not regulation_number:
            regulation_number = self._extract_regulation_number(content)
        
        return DocumentMetadata(
            document_id=document_id,
            title=title,
            document_type=doc_type,
            source_url=source_url,
            regulation_number=regulation_number,
            year=year,
            category=category,
            created_at=datetime.now(),
            content_length=len(content),
            content_hash=content_hash,
            metadata={"file_path": file_path} if file_path else {}
        )
    
    def _detect_category(self, title: str, content: str) -> str:
        """Auto-detect document category based on content."""
        categories = {
            "peraturan_pemerintah": ["peraturan pemerintah", "pp no", "pp nomor"],
            "undang_undang": ["undang-undang", "uu no", "uu nomor"],
            "keputusan_presiden": ["keputusan presiden", "kepres no", "kepres nomor"],
            "peraturan_presiden": ["peraturan presiden", "perpres no", "perpres nomor"],
            "peraturan_menteri": ["peraturan menteri", "permen no", "permen nomor"],
            "keputusan_menteri": ["keputusan menteri", "kepmen no", "kepmen nomor"]
        }
        
        text_lower = f"{title} {content[:1000]}".lower()
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "dokumen_hukum"
    
    def _extract_year(self, content: str) -> Optional[int]:
        """Extract year from document content."""
        # Look for year patterns (4 digits between 1900-2030)
        year_pattern = r'\b(19|20)\d{2}\b'
        matches = re.findall(year_pattern, content)
        
        if matches:
            years = [int(match[0] + match[1:]) for match in matches if match]
            # Return the most recent reasonable year
            valid_years = [year for year in years if 1945 <= year <= 2030]
            if valid_years:
                return max(valid_years)
        
        return None
    
    def _extract_regulation_number(self, content: str) -> Optional[str]:
        """Extract regulation number from content."""
        patterns = [
            r'No\.?\s*(\d+)\s*Tahun\s*(\d{4})',
            r'Nomor\s*(\d+)\s*Tahun\s*(\d{4})',
            r'No\.\s*(\d+/[A-Z]+/\d{4})',
            r'Nomor\s*(\d+/[A-Z]+/\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    async def process_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str,
        title: str,
        doc_type: DocumentType,
        **kwargs
    ) -> List[Document]:
        """Process uploaded file through complete pipeline."""
        try:
            # Extract content based on file type
            if doc_type == DocumentType.PDF:
                extracted_content = await self._extract_pdf_with_ocr(file_content)
            else:
                # For text files
                extracted_content = file_content.decode('utf-8')
            
            if not extracted_content or len(extracted_content.strip()) < 100:
                raise ValueError("Insufficient content extracted from file")
            
            # Create metadata
            metadata = self.create_document_metadata(
                title=title,
                content=extracted_content,
                doc_type=doc_type,
                **kwargs
            )
            
            # Process through pipeline
            documents = await self.process_document_pipeline(extracted_content, metadata)
            
            logger.info(f"✅ File processed: {filename}, {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to process uploaded file {filename}: {e}")
            raise
    
    async def process_predefined_documents(self) -> List[Document]:
        """Process predefined documents from the documents folder."""
        return await self.load_documents_from_folder("documents")