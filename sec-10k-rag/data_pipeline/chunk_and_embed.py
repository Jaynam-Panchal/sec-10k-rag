"""
Chunk documents and generate embeddings.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config import config
from utils.logger import setup_logger
from utils.exceptions import ChunkingError, EmbeddingError
from utils.validation import Chunk, ChunkMetadata, EmbeddingData

logger = setup_logger(__name__, config.LOGS_DIR / "embedding.log")


class TextChunker:
    """
    Chunk text documents into smaller segments with overlap.
    
    Uses semantic chunking based on sentences for better context preservation.
    """
    
    def __init__(
        self, 
        chunk_size: int = None,
        overlap: int = None
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target words per chunk
            overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
        logger.info(
            f"Initialized chunker: chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}"
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        import re
        
        # Split on period, question mark, exclamation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        try:
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                logger.warning("No sentences found in text")
                return []
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                # Start new chunk if size exceeded
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Keep overlap sentences
                    overlap_sentences = []
                    overlap_length = 0
                    
                    for s in reversed(current_chunk):
                        s_len = len(s.split())
                        if overlap_length + s_len <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += s_len
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Create Chunk objects with metadata
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                chunk_meta = ChunkMetadata(
                    chunk_id=f"{metadata.get('filename', 'unknown')}_{i}",
                    ticker=metadata.get('ticker', 'UNKNOWN'),
                    file_name=metadata.get('filename', 'unknown'),
                    chunk_index=i,
                    total_chunks=len(chunks),
                    section=metadata.get('section'),
                    filing_date=metadata.get('filing_date')
                )
                
                chunk_obj = Chunk(text=chunk_text, metadata=chunk_meta)
                chunk_objects.append(chunk_obj)
            
            logger.debug(f"Created {len(chunk_objects)} chunks")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            raise ChunkingError(f"Failed to chunk text: {e}") from e


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using sentence transformers.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully (dim={config.EMBEDDING_DIM})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise EmbeddingError(f"Model loading failed: {e}") from e
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)
        """
        try:
            if not texts:
                return np.array([])
            
            logger.debug(f"Embedding {len(texts)} texts")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=config.BATCH_SIZE,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingData]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of EmbeddingData objects
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Combine with metadata
        embedding_data = []
        for chunk, embedding in zip(chunks, embeddings):
            emb_data = EmbeddingData(
                id=chunk.metadata.chunk_id,
                embedding=embedding.tolist(),
                metadata=chunk.metadata.dict()
            )
            embedding_data.append(emb_data)
        
        return embedding_data


class DocumentProcessor:
    """
    Complete document processing pipeline: chunk + embed.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.processed_count = 0
        self.error_count = 0
    
    def process_file(self, filepath: Path) -> Tuple[List[Chunk], List[EmbeddingData]]:
        """
        Process a single document file.
        
        Args:
            filepath: Path to clean text file
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        try:
            logger.info(f"Processing {filepath.name}")
            
            # Read text
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract metadata from filename
            filename = filepath.stem
            parts = filename.split('_')
            ticker = parts[0] if parts else 'UNKNOWN'
            
            # Load sections if available
            sections_path = filepath.parent / f"{filename}_sections.json"
            sections = {}
            if sections_path.exists():
                with open(sections_path, 'r') as f:
                    sections = json.load(f)
            
            # Chunk text (process each section separately for better context)
            all_chunks = []
            
            if sections and sections != {'FULL_TEXT': text}:
                for section_name, section_text in sections.items():
                    metadata = {
                        'ticker': ticker,
                        'filename': filename,
                        'section': section_name
                    }
                    chunks = self.chunker.chunk_text(section_text, metadata)
                    all_chunks.extend(chunks)
            else:
                # Process full text
                metadata = {
                    'ticker': ticker,
                    'filename': filename,
                    'section': None
                }
                all_chunks = self.chunker.chunk_text(text, metadata)
            
            # Generate embeddings
            embeddings = self.embedder.embed_chunks(all_chunks)
            
            logger.info(
                f"Processed {ticker}: {len(all_chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
            
            self.processed_count += 1
            return all_chunks, embeddings
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)
            self.error_count += 1
            return [], []
    
    def save_embeddings(
        self, 
        chunks: List[Chunk], 
        embeddings: List[EmbeddingData],
        output_dir: Path = None
    ):
        """
        Save chunks and embeddings to disk.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            output_dir: Output directory
        """
        output_dir = output_dir or config.CHUNKS_DIR
        
        for chunk, embedding in zip(chunks, embeddings):
            # Save as individual JSON files for easy loading
            chunk_id = chunk.metadata.chunk_id
            filepath = output_dir / f"{chunk_id}.json"
            
            data = {
                'id': chunk_id,
                'text': chunk.text,
                'embedding': embedding.embedding,
                'meta': embedding.metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)
    
    def process_all(self, parallel: bool = True) -> Dict:
        """
        Process all clean text files.
        
        Args:
            parallel: Use parallel processing
            
        Returns:
            Summary statistics
        """
        files = list(config.CLEAN_TXT_DIR.glob('*.txt'))
        # Filter out summary files
        files = [f for f in files if not f.name.startswith('_')]
        
        logger.info(f"Found {len(files)} files to process")
        
        all_chunks = []
        all_embeddings = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.process_file, f): f.name 
                    for f in files
                }
                
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        chunks, embeddings = future.result()
                        if chunks and embeddings:
                            all_chunks.extend(chunks)
                            all_embeddings.extend(embeddings)
                            # Save immediately
                            self.save_embeddings(chunks, embeddings)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
        else:
            for filepath in files:
                chunks, embeddings = self.process_file(filepath)
                if chunks and embeddings:
                    all_chunks.extend(chunks)
                    all_embeddings.extend(embeddings)
                    self.save_embeddings(chunks, embeddings)
        
        stats = {
            'total_files': len(files),
            'processed_files': self.processed_count,
            'error_files': self.error_count,
            'total_chunks': len(all_chunks),
            'total_embeddings': len(all_embeddings)
        }
        
        logger.info(
            f"Processing complete: {stats['processed_files']} files, "
            f"{stats['total_chunks']} chunks"
        )
        
        return stats


def main():
    """Main execution function."""
    processor = DocumentProcessor()
    
    try:
        stats = processor.process_all(parallel=True)
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete")
        print(f"Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Output directory: {config.CHUNKS_DIR.absolute()}")
        print(f"{'='*60}\n")
        
        # Save summary
        summary_path = config.CHUNKS_DIR / "_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()