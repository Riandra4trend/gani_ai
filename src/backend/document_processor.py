import os
import io
import re
import asyncio
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

# PDF Processing & OCR
import fitz  # PyMuPDF for PDF processing
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Text Processing & NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Indonesian stemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# LangChain & Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

# Models
from models import DocumentMetadata, DocumentType

# Setup logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data")

class DocumentProcessor:
    """Enhanced document processor with OCR and Indonesian NLP pipeline."""
    
    def __init__(self, 
                 documents_folder: str = "./documents",
                 ollama_model: str = "mxbai-embed-large",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            documents_folder: Path to folder containing PDF documents
            ollama_model: Ollama model name for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_folder = Path(documents_folder)
        self.ollama_model = ollama_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # File to track processed documents
        self.processed_files_db = self.documents_folder / "processed_files.json"
        self.processed_files = self._load_processed_files()
        
        # Initialize components
        self._setup_nlp_components()
        self._setup_embeddings()
        self._setup_text_splitter()
        
        # Ensure documents folder exists
        self.documents_folder.mkdir(exist_ok=True)
        
        logger.info(f"✅ DocumentProcessor initialized with Ollama model: {ollama_model}")
    
    def _load_processed_files(self) -> Dict[str, Dict]:
        """Load the database of processed files."""
        try:
            if self.processed_files_db.exists():
                with open(self.processed_files_db, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processed files database: {e}")
        return {}
    
    def _save_processed_files(self):
        """Save the database of processed files."""
        try:
            with open(self.processed_files_db, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save processed files database: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content for change detection."""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.md5(file_content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate hash for {file_path}: {e}")
            return ""
    
    def _is_file_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed and hasn't changed."""
        try:
            file_key = str(file_path.name)
            if file_key not in self.processed_files:
                return False
            
            stored_info = self.processed_files[file_key]
            current_hash = self._get_file_hash(file_path)
            current_size = file_path.stat().st_size
            current_mtime = file_path.stat().st_mtime
            
            # Check if file has changed
            return (
                stored_info.get('hash') == current_hash and
                stored_info.get('size') == current_size and
                stored_info.get('mtime') == current_mtime
            )
        except Exception as e:
            logger.error(f"Error checking if file is processed {file_path}: {e}")
            return False
    
    def _mark_file_processed(self, file_path: Path, document_count: int):
        """Mark file as processed with metadata."""
        try:
            file_key = str(file_path.name)
            self.processed_files[file_key] = {
                'hash': self._get_file_hash(file_path),
                'size': file_path.stat().st_size,
                'mtime': file_path.stat().st_mtime,
                'processed_at': datetime.now().isoformat(),
                'document_count': document_count,
                'file_path': str(file_path)
            }
            self._save_processed_files()
        except Exception as e:
            logger.error(f"Failed to mark file as processed {file_path}: {e}")
    
    def get_new_files(self) -> List[Path]:
        """Get list of new or modified files that need processing."""
        pdf_files = list(self.documents_folder.glob("*.pdf"))
        new_files = []
        
        for pdf_file in pdf_files:
            if not self._is_file_processed(pdf_file):
                new_files.append(pdf_file)
                logger.info(f"📄 New/modified file detected: {pdf_file.name}")
        
        return new_files
    
    def get_processed_files_info(self) -> Dict[str, Any]:
        """Get information about processed files."""
        total_files = len(list(self.documents_folder.glob("*.pdf")))
        processed_count = len(self.processed_files)
        total_chunks = sum(info.get('document_count', 0) for info in self.processed_files.values())
        
        return {
            "total_pdf_files": total_files,
            "processed_files": processed_count,
            "unprocessed_files": total_files - processed_count,
            "total_chunks_created": total_chunks,
            "processed_files_list": list(self.processed_files.keys())
        }
    
    def _setup_nlp_components(self):
        """Setup Indonesian NLP components."""
        try:
            # Indonesian stemmer
            stemmer_factory = StemmerFactory()
            self.stemmer = stemmer_factory.create_stemmer()
            
            # Indonesian stopword remover
            stopword_factory = StopWordRemoverFactory()
            self.stopword_remover = stopword_factory.create_stop_word_remover()
            
            # Additional Indonesian stopwords
            indonesian_stopwords = set(stopwords.words('indonesian')) if 'indonesian' in stopwords.fileids() else set()
            custom_stopwords = {
                'yang', 'adalah', 'dengan', 'untuk', 'pada', 'dalam', 'dari', 
                'oleh', 'akan', 'telah', 'sudah', 'dapat', 'harus', 'bisa',
                'pasal', 'ayat', 'undang', 'peraturan', 'pemerintah'
            }
            self.stopwords = indonesian_stopwords.union(custom_stopwords)
            
            logger.info("✅ Indonesian NLP components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup NLP components: {e}")
            # Fallback to basic processing
            self.stemmer = None
            self.stopword_remover = None
            self.stopwords = set()
    
    def _setup_embeddings(self):
        """Setup Ollama embeddings."""
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.ollama_model,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
            logger.info(f"✅ Ollama embeddings initialized: {self.ollama_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Ollama embeddings: {e}")
            raise
    
    def _setup_text_splitter(self):
        """Setup text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    async def load_new_documents_only(self) -> List[Document]:
        """
        Load and process only new or modified documents from the documents folder.
        
        Returns:
            List of processed Document objects from new files
        """
        logger.info(f"🔄 Checking for new documents in: {self.documents_folder}")
        
        new_files = self.get_new_files()
        if not new_files:
            logger.info("✅ No new documents to process")
            return []
        
        logger.info(f"📁 Found {len(new_files)} new/modified files to process")
        
        all_documents = []
        
        for pdf_file in new_files:
            try:
                logger.info(f"📖 Processing: {pdf_file.name}")
                
                # Extract text from PDF
                text_content = await self._extract_pdf_content(pdf_file)
                
                if not text_content or len(text_content.strip()) < 100:
                    logger.warning(f"⚠️ Insufficient content in {pdf_file.name}")
                    continue
                
                # Create document metadata
                metadata = self._create_document_metadata(
                    title=pdf_file.stem,
                    content=text_content,
                    doc_type=DocumentType.PDF,
                    source_file=str(pdf_file)
                )
                
                # Process document through pipeline
                processed_docs = await self._process_document_pipeline(
                    content=text_content,
                    metadata=metadata
                )
                
                all_documents.extend(processed_docs)
                
                # Mark file as processed
                self._mark_file_processed(pdf_file, len(processed_docs))
                
                logger.info(f"✅ Processed {pdf_file.name}: {len(processed_docs)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✅ Total new documents processed: {len(all_documents)}")
        return all_documents
    
    async def load_documents_from_folder(self) -> List[Document]:
        """
        Load and process all PDF documents from the documents folder.
        
        Returns:
            List of processed Document objects
        """
        logger.info(f"🔄 Loading documents from: {self.documents_folder}")
        
        pdf_files = list(self.documents_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {self.documents_folder}")
            return []
        
        logger.info(f"📁 Found {len(pdf_files)} PDF files")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"📖 Processing: {pdf_file.name}")
                
                # Extract text from PDF
                text_content = await self._extract_pdf_content(pdf_file)
                
                if not text_content or len(text_content.strip()) < 100:
                    logger.warning(f"⚠️ Insufficient content in {pdf_file.name}")
                    continue
                
                # Create document metadata
                metadata = self._create_document_metadata(
                    title=pdf_file.stem,
                    content=text_content,
                    doc_type=DocumentType.PDF,
                    source_file=str(pdf_file)
                )
                
                # Process document through pipeline
                processed_docs = await self._process_document_pipeline(
                    content=text_content,
                    metadata=metadata
                )
                
                all_documents.extend(processed_docs)
                
                # Mark file as processed
                self._mark_file_processed(pdf_file, len(processed_docs))
                
                logger.info(f"✅ Processed {pdf_file.name}: {len(processed_docs)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✅ Total processed documents: {len(all_documents)}")
        return all_documents
    
    async def force_reprocess_all(self) -> List[Document]:
        """
        Force reprocess all documents, ignoring the processed files database.
        
        Returns:
            List of processed Document objects
        """
        logger.info("🔄 Force reprocessing all documents...")
        
        # Clear processed files database
        self.processed_files = {}
        self._save_processed_files()
        
        # Process all documents
        return await self.load_documents_from_folder()
    
    def clear_processed_files_db(self):
        """Clear the processed files database."""
        self.processed_files = {}
        self._save_processed_files()
        logger.info("🗑️ Cleared processed files database")
    
    async def _extract_pdf_content(self, pdf_path: Path) -> str:
        """
        Extract text content from PDF using PyMuPDF + OCR fallback.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # First try PyMuPDF for text extraction
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # If no text found, use OCR
                if not page_text.strip():
                    logger.info(f"🔍 Using OCR for page {page_num + 1} of {pdf_path.name}")
                    page_text = await self._ocr_page(page)
                
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            return text_content
            
        except Exception as e:
            logger.error(f"❌ PDF extraction failed for {pdf_path}: {e}")
            return ""
    
    async def _ocr_page(self, page) -> str:
        """
        Perform OCR on a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            OCR extracted text
        """
        try:
            # Get page as image
            mat = fitz.Matrix(2, 2)  # Scale factor for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.pil_tobytes(format="PNG")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to OpenCV format for preprocessing
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(opencv_image)
            
            # Perform OCR with Indonesian language
            ocr_text = pytesseract.image_to_string(
                processed_image,
                lang='ind+eng',  # Indonesian + English
                config='--oem 3 --psm 6'  # OCR Engine Mode and Page Segmentation Mode
            )
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    async def _process_document_pipeline(self, content: str, metadata: DocumentMetadata) -> List[Document]:
        """
        Complete document processing pipeline.
        
        Args:
            content: Raw text content
            metadata: Document metadata
            
        Returns:
            List of processed Document chunks
        """
        # Step 1: Clean text
        cleaned_text = self._clean_text(content)
        
        # Step 2: Tokenize and preprocess
        preprocessed_text = self._preprocess_text(cleaned_text)
        
        # Step 3: Chunk the text
        chunks = self.text_splitter.split_text(preprocessed_text)
        
        # Step 4: Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            # Create chunk metadata
            chunk_metadata = {
                "document_id": metadata.document_id,
                "title": metadata.title,
                "chunk_id": f"{metadata.document_id}_{i}",
                "chunk_index": i,
                "source": getattr(metadata, "source_file", getattr(metadata, "source", None)),
                "document_type": metadata.document_type.value if hasattr(metadata.document_type, "value") else str(metadata.document_type),
                "created_at": metadata.created_at.isoformat() if getattr(metadata, "created_at", None) else None,
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            }
            
            # Add optional metadata
            if metadata.regulation_number:
                chunk_metadata["regulation_number"] = metadata.regulation_number
            if metadata.year:
                chunk_metadata["year"] = metadata.year
            if metadata.category:
                chunk_metadata["category"] = metadata.category
            
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        Clean raw text content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove page markers
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', ' ', text)
        
        # Remove numbers that are likely page numbers or formatting artifacts
        text = re.sub(r'\b\d{1,3}\b', '', text)
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text with Indonesian NLP pipeline.
        
        Args:
            text: Cleaned text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove stopwords if available
        if self.stopword_remover:
            text = self.stopword_remover.remove(text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text, language='indonesian')
        except:
            tokens = text.split()
        
        # Remove stopwords manually if NLTK failed
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Stem words if stemmer available
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens if len(token) > 2]
        else:
            tokens = [token for token in tokens if len(token) > 2]
        
        # Rejoin tokens
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def _create_document_metadata(self, 
                                  title: str, 
                                  content: str, 
                                  doc_type: DocumentType,
                                  source_file: str = None,
                                  **kwargs) -> DocumentMetadata:
        """
        Create document metadata.
        
        Args:
            title: Document title
            content: Document content
            doc_type: Document type
            source_file: Source file path
            **kwargs: Additional metadata
            
        Returns:
            DocumentMetadata object
        """
        return DocumentMetadata(
            document_id=str(uuid4()),
            title=title,
            document_type=doc_type,
            content_length=len(content),
            created_at=datetime.now(),
            regulation_number=kwargs.get('regulation_number'),
            year=kwargs.get('year'),
            category=kwargs.get('category'),
            source_url=kwargs.get('source_url'),
            # fallback supaya tetap ada info sumber
            source_file=source_file if hasattr(DocumentMetadata, "source_file") else None  
        )
    
    async def process_uploaded_file(self, 
                                    file_content: bytes, 
                                    filename: str,
                                    title: str,
                                    **kwargs) -> List[Document]:
        """
        Process uploaded PDF file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            title: Document title
            **kwargs: Additional metadata
            
        Returns:
            List of processed Document chunks
        """
        logger.info(f"🔄 Processing uploaded file: {filename}")
        
        try:
            # Extract text from PDF bytes
            text_content = await self._extract_pdf_from_bytes(file_content)
            
            if not text_content or len(text_content.strip()) < 100:
                raise ValueError("Insufficient content extracted from PDF")
            
            # Create metadata
            metadata = self._create_document_metadata(
                title=title,
                content=text_content,
                doc_type=DocumentType.PDF,
                source_file=filename,
                **kwargs
            )
            
            # Process through pipeline
            documents = await self._process_document_pipeline(text_content, metadata)
            
            logger.info(f"✅ Processed {filename}: {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to process uploaded file {filename}: {e}")
            raise
    
    async def _extract_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text content
        """
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # If no text found, use OCR
                if not page_text.strip():
                    logger.info(f"🔍 Using OCR for page {page_num + 1}")
                    page_text = await self._ocr_page(page)
                
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            return text_content
            
        except Exception as e:
            logger.error(f"❌ PDF bytes extraction failed: {e}")
            raise
    
    async def create_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Create embeddings for documents using Ollama.
        
        Args:
            documents: List of documents
            
        Returns:
            Documents with embeddings
        """
        logger.info(f"🔄 Creating embeddings for {len(documents)} documents...")
        
        try:
            # Extract texts for embedding
            texts = [doc.page_content for doc in documents]
            
            # Create embeddings using Ollama
            embeddings_list = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings_list):
                doc.metadata["embedding"] = embedding
            
            logger.info(f"✅ Created embeddings for {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        files_info = self.get_processed_files_info()
        
        return {
            "documents_folder": str(self.documents_folder),
            "ollama_model": self.ollama_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_info": files_info,
            "nlp_components": {
                "stemmer_available": self.stemmer is not None,
                "stopword_remover_available": self.stopword_remover is not None,
                "stopwords_count": len(self.stopwords)
            }
        }