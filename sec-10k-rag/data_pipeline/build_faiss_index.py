"""
Build FAISS vector index for semantic search.
"""
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple

from config.config import config
from utils.logger import setup_logger
from utils.exceptions import IndexError as IndexBuildError

logger = setup_logger(__name__, config.LOGS_DIR / "index.log")


class FAISSIndexBuilder:
    """
    Build and manage FAISS vector index for similarity search.
    
    Supports:
    - Multiple index types (Flat, IVF, HNSW)
    - Metadata storage
    - Index serialization
    """
    
    def __init__(self, index_type: str = None):
        """
        Initialize index builder.
        
        Args:
            index_type: Type of FAISS index to build
        """
        self.index_type = index_type or config.FAISS_INDEX_TYPE
        self.index = None
        self.embeddings = []
        self.ids = []
        self.metadatas = []
        
        logger.info(f"Initialized FAISS index builder: {self.index_type}")
    
    def load_embeddings(self, chunks_dir: Path = None) -> Tuple[np.ndarray, List, List]:
        """
        Load all embeddings from chunk files.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
            
        Returns:
            Tuple of (embeddings array, ids list, metadatas list)
        """
        chunks_dir = chunks_dir or config.CHUNKS_DIR
        
        logger.info(f"Loading embeddings from {chunks_dir}")
        
        embeddings = []
        ids = []
        metadatas = []
        
        # Get all chunk files
        chunk_files = sorted(chunks_dir.glob("*.json"))
        # Filter out summary files
        chunk_files = [f for f in chunk_files if not f.name.startswith('_')]
        
        if not chunk_files:
            raise IndexBuildError(
                f"No chunk files found in {chunks_dir}. "
                "Run chunk_and_embed.py first."
            )
        
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Load each chunk
        for filepath in chunk_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                embeddings.append(data['embedding'])
                ids.append(data['id'])
                metadatas.append(data['meta'])
                
            except Exception as e:
                logger.warning(f"Error loading {filepath.name}: {e}")
                continue
        
        if not embeddings:
            raise IndexBuildError("No valid embeddings loaded")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        logger.info(
            f"Loaded {len(embeddings)} embeddings "
            f"(shape: {embeddings_array.shape})"
        )
        
        return embeddings_array, ids, metadatas
    
    def build_index(
        self, 
        embeddings: np.ndarray,
        normalize: bool = True
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Embedding vectors (n_vectors x dim)
            normalize: Normalize vectors for cosine similarity
            
        Returns:
            FAISS index
        """
        logger.info(f"Building {self.index_type} index...")
        
        n_vectors, dim = embeddings.shape
        
        if n_vectors == 0:
            raise IndexBuildError("Cannot build index with 0 vectors")
        
        # Normalize embeddings for cosine similarity
        if normalize:
            logger.debug("Normalizing embeddings (L2)")
            faiss.normalize_L2(embeddings)
        
        # Build index based on type
        if self.index_type == "IndexFlatIP":
            # Exact search with inner product (cosine similarity if normalized)
            index = faiss.IndexFlatIP(dim)
            
        elif self.index_type == "IndexFlatL2":
            # Exact search with L2 distance
            index = faiss.IndexFlatL2(dim)
            
        elif self.index_type == "IndexIVFFlat":
            # Faster approximate search with IVF
            nlist = min(100, n_vectors // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train index
            logger.info("Training IVF index...")
            index.train(embeddings)
            
        elif self.index_type == "IndexHNSWFlat":
            # HNSW for fast approximate search
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 = number of connections
            
        else:
            raise IndexBuildError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        logger.info(f"Adding {n_vectors} vectors to index...")
        index.add(embeddings)
        
        logger.info(
            f"Index built successfully: {index.ntotal} vectors, "
            f"dim={dim}, type={self.index_type}"
        )
        
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        ids: List[str],
        metadatas: List[Dict],
        output_dir: Path = None
    ):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index
            ids: List of chunk IDs
            metadatas: List of metadata dicts
            output_dir: Output directory
        """
        output_dir = output_dir or config.INDEX_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving index to {output_dir}")
        
        # Save FAISS index
        index_path = output_dir / "faiss.index"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index: {index_path}")
        
        # Save IDs
        ids_path = output_dir / "ids.pkl"
        with open(ids_path, 'wb') as f:
            pickle.dump(ids, f)
        logger.info(f"Saved IDs: {ids_path}")
        
        # Save metadata
        metadata_path = output_dir / "metadatas.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadatas, f)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Save index info
        info = {
            'index_type': self.index_type,
            'n_vectors': index.ntotal,
            'dimension': index.d,
            'ids_count': len(ids),
            'metadata_count': len(metadatas)
        }
        
        info_path = output_dir / "index_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved index info: {info_path}")
    
    def build_and_save(self):
        """
        Complete pipeline: load embeddings, build index, save.
        """
        try:
            # Load embeddings
            embeddings, ids, metadatas = self.load_embeddings()
            
            # Store for later use
            self.embeddings = embeddings
            self.ids = ids
            self.metadatas = metadatas
            
            # Build index
            self.index = self.build_index(embeddings)
            
            # Save everything
            self.save_index(self.index, ids, metadatas)
            
            logger.info("Index building complete!")
            
            return {
                'n_vectors': len(embeddings),
                'dimension': embeddings.shape[1],
                'index_type': self.index_type
            }
            
        except Exception as e:
            logger.error(f"Index building failed: {e}", exc_info=True)
            raise


class FAISSIndexLoader:
    """
    Load and use a saved FAISS index.
    """
    
    def __init__(self, index_dir: Path = None):
        """
        Initialize loader.
        
        Args:
            index_dir: Directory containing saved index
        """
        self.index_dir = index_dir or config.INDEX_DIR
        self.index = None
        self.ids = None
        self.metadatas = None
        
    def load(self):
        """Load index and metadata from disk."""
        logger.info(f"Loading index from {self.index_dir}")
        
        # Load FAISS index
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise IndexBuildError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load IDs
        ids_path = self.index_dir / "ids.pkl"
        with open(ids_path, 'rb') as f:
            self.ids = pickle.load(f)
        
        # Load metadata
        metadata_path = self.index_dir / "metadatas.pkl"
        with open(metadata_path, 'rb') as f:
            self.metadatas = pickle.load(f)
        
        logger.info("Index loaded successfully")
        
    def search(
        self, 
        query_vector: np.ndarray,
        top_k: int = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search index for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadatas)
        """
        if self.index is None:
            raise IndexBuildError("Index not loaded. Call load() first.")
        
        top_k = top_k or config.TOP_K_RESULTS
        
        # Normalize query
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get metadata for results
        result_metadata = [
            self.metadatas[idx] for idx in indices[0] if idx < len(self.metadatas)
        ]
        
        return distances[0], indices[0], result_metadata


def main():
    """Main execution function."""
    builder = FAISSIndexBuilder()
    
    try:
        stats = builder.build_and_save()
        
        print(f"\n{'='*60}")
        print(f"✓ FAISS index built successfully")
        print(f"Vectors: {stats['n_vectors']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Index type: {stats['index_type']}")
        print(f"Saved to: {config.INDEX_DIR.absolute()}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()