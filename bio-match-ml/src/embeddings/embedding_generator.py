"""
Embedding Generator for creating vector representations of text
"""
import logging
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Multi-model embedding generator with caching and optimization
    """

    def __init__(self, model_manager=None, config_path: Optional[str] = None):
        """
        Initialize EmbeddingGenerator

        Args:
            model_manager: ModelManager instance (will create if None)
            config_path: Path to configuration file
        """
        if model_manager is None:
            from .model_manager import ModelManager
            self.model_manager = ModelManager(config_path)
        else:
            self.model_manager = model_manager

        self.device = self.model_manager.device
        self.cache = None  # Will be set via set_cache method

        logger.info("EmbeddingGenerator initialized")

    def set_cache(self, cache_manager):
        """Set cache manager for embedding caching"""
        self.cache = cache_manager

    def generate_embedding(
        self,
        text: str,
        model_name: Optional[str] = None,
        pooling_strategy: str = 'mean',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Input text
            model_name: Name of model to use (None = auto-select)
            pooling_strategy: How to pool token embeddings (mean, max, cls)
            normalize: Whether to L2-normalize the embedding

        Returns:
            768-dimensional embedding vector
        """
        # Check cache first
        if self.cache is not None:
            cache_key = f"{model_name}:{pooling_strategy}:{hash(text)}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Auto-select model if not specified
        if model_name is None:
            model_name = self.model_manager.auto_select_model(text)

        # Get model and tokenizer
        model, tokenizer, config = self.model_manager.get_model(model_name)

        # Tokenize
        max_length = config.get('max_length', 512)
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )

        # Move to device
        if self.device.type == 'cuda':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Pool token embeddings
        if pooling_strategy == 'mean':
            # Mean pooling over all tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif pooling_strategy == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            embedding = torch.max(token_embeddings, dim=1)[0]
        elif pooling_strategy == 'cls':
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Convert to numpy
        embedding = embedding.cpu().numpy().flatten()

        # Normalize
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        # Cache result
        if self.cache is not None:
            cache_key = f"{model_name}:{pooling_strategy}:{hash(text)}"
            self.cache.set(cache_key, embedding)

        return embedding

    def batch_generate(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        pooling_strategy: str = 'mean',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Efficient batch processing with GPU utilization

        Args:
            texts: List of input texts
            model_name: Name of model to use
            batch_size: Batch size for processing
            show_progress: Show progress bar
            pooling_strategy: Pooling method
            normalize: Normalize embeddings

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        # Auto-select model if not specified
        if model_name is None:
            model_name = self.model_manager.auto_select_model(texts[0] if texts else "")

        # Get model and tokenizer
        model, tokenizer, config = self.model_manager.get_model(model_name)
        max_length = config.get('max_length', 512)

        embeddings = []

        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating embeddings ({model_name})")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Check cache for batch items
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []

            if self.cache is not None:
                for idx, text in enumerate(batch_texts):
                    cache_key = f"{model_name}:{pooling_strategy}:{hash(text)}"
                    cached = self.cache.get(cache_key)
                    if cached is not None:
                        cached_embeddings[idx] = cached
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(idx)
            else:
                uncached_texts = batch_texts
                uncached_indices = list(range(len(batch_texts)))

            # Generate embeddings for uncached texts
            batch_embeddings = [None] * len(batch_texts)

            if uncached_texts:
                # Tokenize batch
                inputs = tokenizer(
                    uncached_texts,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )

                # Move to device
                if self.device.type == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)

                # Pool token embeddings
                if pooling_strategy == 'mean':
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                        token_embeddings.size()
                    ).float()
                    batch_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9
                    )
                elif pooling_strategy == 'max':
                    token_embeddings = outputs.last_hidden_state
                    batch_emb = torch.max(token_embeddings, dim=1)[0]
                elif pooling_strategy == 'cls':
                    batch_emb = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

                # Convert to numpy
                batch_emb = batch_emb.cpu().numpy()

                # Normalize if requested
                if normalize:
                    norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                    batch_emb = batch_emb / norms

                # Place in batch_embeddings and cache
                for idx, emb_idx in enumerate(uncached_indices):
                    embedding = batch_emb[idx]
                    batch_embeddings[emb_idx] = embedding

                    # Cache the embedding
                    if self.cache is not None:
                        text = uncached_texts[idx]
                        cache_key = f"{model_name}:{pooling_strategy}:{hash(text)}"
                        self.cache.set(cache_key, embedding)

            # Fill in cached embeddings
            for idx, embedding in cached_embeddings.items():
                batch_embeddings[idx] = embedding

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def generate_hierarchical_embedding(
        self,
        document: Dict[str, str],
        model_name: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Create multi-level embeddings from document sections

        Args:
            document: Dict with keys like 'title', 'abstract', 'sections'
            model_name: Model to use
            weights: Weight for each section

        Returns:
            Weighted combined embedding
        """
        default_weights = {
            'title': 0.3,
            'abstract': 0.5,
            'sections': 0.2
        }
        weights = weights or default_weights

        embeddings = []
        section_weights = []

        for section, text in document.items():
            if text and section in weights:
                emb = self.generate_embedding(text, model_name=model_name)
                embeddings.append(emb)
                section_weights.append(weights[section])

        if not embeddings:
            raise ValueError("No valid sections found in document")

        # Weighted combination
        embeddings = np.array(embeddings)
        section_weights = np.array(section_weights).reshape(-1, 1)

        # Normalize weights
        section_weights = section_weights / section_weights.sum()

        # Combine
        combined = np.sum(embeddings * section_weights, axis=0)

        # Normalize
        combined = combined / np.linalg.norm(combined)

        return combined

    def generate_multimodal_embedding(
        self,
        profile: Dict[str, any],
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Combine multiple signal types from a faculty profile

        Args:
            profile: Faculty profile dict with various fields
            model_name: Model to use

        Returns:
            Combined embedding representing entire profile
        """
        embeddings = []
        weights = []

        # Research summary/interests (high weight)
        if 'research_summary' in profile and profile['research_summary']:
            emb = self.generate_embedding(profile['research_summary'], model_name=model_name)
            embeddings.append(emb)
            weights.append(0.4)

        # Recent publications (medium-high weight)
        if 'recent_publications' in profile and profile['recent_publications']:
            pub_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in profile['recent_publications'][:5]  # Top 5 recent
            ]
            if pub_texts:
                pub_embeddings = self.batch_generate(
                    pub_texts,
                    model_name=model_name,
                    show_progress=False
                )
                # Average publication embeddings
                avg_pub_emb = np.mean(pub_embeddings, axis=0)
                embeddings.append(avg_pub_emb)
                weights.append(0.3)

        # Grant abstracts (medium weight)
        if 'active_grants' in profile and profile['active_grants']:
            grant_texts = [
                g.get('abstract', '')
                for g in profile['active_grants']
                if g.get('abstract')
            ]
            if grant_texts:
                grant_embeddings = self.batch_generate(
                    grant_texts,
                    model_name=model_name,
                    show_progress=False
                )
                avg_grant_emb = np.mean(grant_embeddings, axis=0)
                embeddings.append(avg_grant_emb)
                weights.append(0.2)

        # Techniques and keywords (lower weight)
        if 'techniques' in profile and profile['techniques']:
            tech_text = ' '.join(profile['techniques'])
            emb = self.generate_embedding(tech_text, model_name=model_name)
            embeddings.append(emb)
            weights.append(0.1)

        if not embeddings:
            raise ValueError("No valid content found in profile")

        # Weighted combination
        embeddings = np.array(embeddings)
        weights = np.array(weights).reshape(-1, 1)

        # Normalize weights
        weights = weights / weights.sum()

        # Combine
        combined = np.sum(embeddings * weights, axis=0)

        # Normalize
        combined = combined / np.linalg.norm(combined)

        return combined

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric (cosine, euclidean, dot)

        Returns:
            Similarity score
        """
        if metric == 'cosine':
            return float(np.dot(embedding1, embedding2))
        elif metric == 'euclidean':
            return float(1.0 / (1.0 + np.linalg.norm(embedding1 - embedding2)))
        elif metric == 'dot':
            return float(np.dot(embedding1, embedding2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Calculate similarity between one query and many corpus embeddings

        Args:
            query_embedding: Query embedding (1D array)
            corpus_embeddings: Corpus embeddings (2D array)
            metric: Similarity metric

        Returns:
            Array of similarity scores
        """
        if metric == 'cosine':
            # Cosine similarity (assuming normalized embeddings)
            similarities = np.dot(corpus_embeddings, query_embedding)
        elif metric == 'euclidean':
            distances = np.linalg.norm(corpus_embeddings - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
        elif metric == 'dot':
            similarities = np.dot(corpus_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return similarities
