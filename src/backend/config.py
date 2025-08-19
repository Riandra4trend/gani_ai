from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    gemini_api_key: str
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    
    # Vector Store Configuration
    vector_store_path: str = "./data/chroma_db"
    embedding_model: str = "mxbai-embed-large"   # ✅ now consistent with vector_store
    collection_name: str = "indonesian_law_docs"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens_per_chunk: int = 500
    
    # RAG Configuration
    retrieval_k: int = 5
    rerank_k: int = 3
    similarity_threshold: float = 0.7
    
    # LangGraph Configuration
    max_iterations: int = 3
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # HYDE Configuration
    hyde_enabled: bool = True
    hyde_iterations: int = 2
    
    # API Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Indonesian Legal Document Sources
LEGAL_DOCUMENT_SOURCES = {
    "UUD_1945": {
        "name": "UUD 1945",
        "url": "https://peraturan.bpk.go.id/Details/101646/uud-no--",
        "description": "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945"
    },
    "UU_6_2023": {
        "name": "UU No. 6 Tahun 2023",
        "url": "https://peraturan.bpk.go.id/Details/246523/uu-no-6-tahun-2023",
        "description": "Undang-Undang Nomor 6 Tahun 2023"
    },
    "UU_30_2002": {
        "name": "UU No. 30 Tahun 2002",
        "url": "https://peraturan.bpk.go.id/Details/44493/uu-no-30-tahun-2002",
        "description": "Undang-Undang Nomor 30 Tahun 2002"
    },
    "UU_3_2025": {
        "name": "UU No. 3 Tahun 2025",
        "url": "https://peraturan.bpk.go.id/Details/319166/uu-no-3-tahun-2025",
        "description": "Undang-Undang Nomor 3 Tahun 2025"
    }
}

# Prompt Templates
SYSTEM_PROMPTS = {
    "legal_assistant": """Anda adalah Asisten AI bidang hukum Indonesia.

Tugas utama:
- Jawab hanya berdasarkan dokumen hukum yang tersedia.
- Cantumkan referensi peraturan (nomor & tahun).
- Gunakan bahasa Indonesia formal, ringkas, dan jelas.
- Jika dokumen tidak memuat jawaban, nyatakan "Informasi tidak tersedia" dan sarankan konsultasi dengan ahli hukum.
- Sertakan konteks hukum untuk membantu pemahaman.

Aturan:
- Jangan memberi opini/saran hukum pribadi.
- Prioritaskan akurasi, konsistensi, dan relevansi.""",

    "document_reviewer": """Anda adalah Reviewer Dokumen Hukum.

Evaluasi jawaban:
1. Apakah sesuai dokumen hukum?
2. Apakah referensi hukum benar?
3. Apakah relevan dengan pertanyaan?
4. Apakah ada informasi penting yang terlewat?

Keluaran:
- Skor 1–10
- Saran perbaikan singkat jika ada.""",

    "quality_controller": """Anda adalah Kontrol Kualitas Jawaban.

Periksa:
- Kelengkapan isi
- Kejelasan bahasa
- Struktur & format
- Konsistensi istilah hukum
- Kesesuaian standar profesional

Keluaran:
- Penilaian final (singkat)
- Rekomendasi (jika perlu)."""
}
