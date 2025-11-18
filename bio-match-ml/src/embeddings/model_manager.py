"""
Model Manager for handling multiple embedding models with optimization
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import yaml
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage multiple embedding models with optimization and benchmarking
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ModelManager with configuration

        Args:
            config_path: Path to model configuration YAML file
        """
        self.config_path = config_path or "configs/model_configs.yaml"
        self.config = self._load_config()

        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"ModelManager initialized on device: {self.device}")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'embedding_models': {
                'pubmedbert': {
                    'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'max_length': 512,
                    'batch_size': 32,
                    'use_gpu': True,
                    'pooling_strategy': 'mean'
                }
            },
            'default': 'pubmedbert'
        }

    def register_model(
        self,
        name: str,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict] = None
    ) -> None:
        """
        Register a new model for use

        Args:
            name: Unique identifier for the model
            model: Pre-loaded model instance
            tokenizer: Pre-loaded tokenizer instance
            config: Model configuration dict
        """
        if name in self.model_registry:
            logger.warning(f"Model '{name}' already registered. Overwriting.")

        # Use provided config or get from main config
        model_config = config or self.config.get('embedding_models', {}).get(name, {})

        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            model_name = model_config.get('model_name', name)
            logger.info(f"Loading model: {model_name}")

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            model = model or AutoModel.from_pretrained(model_name)

            # Move to device
            if model_config.get('use_gpu', True) and self.device.type == 'cuda':
                model = model.to(self.device)

            # Set to eval mode
            model.eval()

        self.model_registry[name] = {
            'model': model,
            'tokenizer': tokenizer,
            'config': model_config,
            'optimized': False
        }

        logger.info(f"Model '{name}' registered successfully")

    def get_model(self, name: str) -> Tuple[Any, Any, Dict]:
        """
        Get model, tokenizer, and config by name

        Args:
            name: Model identifier

        Returns:
            Tuple of (model, tokenizer, config)
        """
        if name not in self.model_registry:
            # Try to auto-register from config
            if name in self.config.get('embedding_models', {}):
                self.register_model(name)
            else:
                raise ValueError(f"Model '{name}' not found in registry")

        model_data = self.model_registry[name]
        return model_data['model'], model_data['tokenizer'], model_data['config']

    def optimize_model(
        self,
        model_name: str,
        use_onnx: bool = False,
        use_quantization: bool = False,
        use_torchscript: bool = False
    ) -> None:
        """
        Apply optimizations to a model

        Args:
            model_name: Name of model to optimize
            use_onnx: Convert to ONNX format
            use_quantization: Apply INT8 quantization
            use_torchscript: Compile with TorchScript
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found")

        model_data = self.model_registry[model_name]
        model = model_data['model']

        logger.info(f"Optimizing model: {model_name}")

        # Apply TorchScript compilation
        if use_torchscript:
            try:
                logger.info("Applying TorchScript compilation...")
                model = torch.jit.script(model)
                model_data['model'] = model
                logger.info("TorchScript compilation successful")
            except Exception as e:
                logger.error(f"TorchScript compilation failed: {e}")

        # Apply quantization
        if use_quantization:
            try:
                logger.info("Applying INT8 quantization...")
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                model_data['model'] = model
                logger.info("Quantization successful")
            except Exception as e:
                logger.error(f"Quantization failed: {e}")

        # ONNX conversion would go here
        if use_onnx:
            logger.info("ONNX conversion not yet implemented")

        model_data['optimized'] = True
        logger.info(f"Model '{model_name}' optimization complete")

    def benchmark_models(
        self,
        test_texts: List[str],
        models_to_test: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple models on test data

        Args:
            test_texts: List of texts to use for benchmarking
            models_to_test: List of model names to benchmark (None = all)

        Returns:
            Dictionary of benchmark results per model
        """
        import time

        models_to_test = models_to_test or list(self.model_registry.keys())
        results = {}

        logger.info(f"Benchmarking {len(models_to_test)} models on {len(test_texts)} texts")

        for model_name in models_to_test:
            if model_name not in self.model_registry:
                logger.warning(f"Model '{model_name}' not in registry, skipping")
                continue

            model, tokenizer, config = self.get_model(model_name)

            # Measure inference time
            start_time = time.time()

            try:
                with torch.no_grad():
                    for text in test_texts:
                        inputs = tokenizer(
                            text,
                            return_tensors='pt',
                            max_length=config.get('max_length', 512),
                            truncation=True,
                            padding=True
                        )

                        if self.device.type == 'cuda':
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        outputs = model(**inputs)

                elapsed = time.time() - start_time

                # Calculate metrics
                texts_per_second = len(test_texts) / elapsed if elapsed > 0 else 0
                avg_time_ms = (elapsed / len(test_texts)) * 1000

                # Estimate memory usage
                if self.device.type == 'cuda':
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    memory_mb = 0

                results[model_name] = {
                    'total_time_s': elapsed,
                    'texts_per_second': texts_per_second,
                    'avg_time_ms': avg_time_ms,
                    'memory_mb': memory_mb,
                    'device': str(self.device)
                }

                # Store in performance metrics
                self.performance_metrics[model_name].append(results[model_name])

                logger.info(
                    f"{model_name}: {texts_per_second:.2f} texts/sec, "
                    f"{avg_time_ms:.2f}ms avg, {memory_mb:.1f}MB memory"
                )

            except Exception as e:
                logger.error(f"Benchmark failed for '{model_name}': {e}")
                results[model_name] = {'error': str(e)}

        return results

    def auto_select_model(
        self,
        text: str,
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Automatically select best model based on text characteristics

        Args:
            text: Input text to analyze
            available_models: List of models to consider (None = all)

        Returns:
            Name of selected model
        """
        available_models = available_models or list(self.model_registry.keys())

        # Analyze text characteristics
        text_length = len(text.split())

        # Simple heuristic: use faster model for short texts,
        # more accurate model for long/complex texts
        if text_length < 50:
            # Prefer faster model for short texts
            if 'biobert' in available_models:
                return 'biobert'
        else:
            # Prefer accuracy for longer texts
            if 'pubmedbert' in available_models:
                return 'pubmedbert'

        # Default to first available or configured default
        default_model = self.config.get('default', 'pubmedbert')
        if default_model in available_models:
            return default_model

        return available_models[0] if available_models else 'pubmedbert'

    def ensemble_embeddings(
        self,
        text: str,
        models: List[str],
        weights: Optional[List[float]] = None,
        fusion_method: str = 'weighted_average'
    ) -> np.ndarray:
        """
        Combine embeddings from multiple models

        Args:
            text: Input text
            models: List of model names to use
            weights: Weights for each model (None = equal weights)
            fusion_method: How to combine embeddings

        Returns:
            Combined embedding vector
        """
        from .embedding_generator import EmbeddingGenerator

        # Initialize generator
        generator = EmbeddingGenerator(model_manager=self)

        # Get embeddings from each model
        embeddings = []
        for model_name in models:
            emb = generator.generate_embedding(text, model_name=model_name)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        # Apply fusion
        if fusion_method == 'weighted_average':
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            weights = np.array(weights).reshape(-1, 1)
            result = np.sum(embeddings * weights, axis=0)

        elif fusion_method == 'concatenation':
            result = np.concatenate(embeddings)

        elif fusion_method == 'max':
            result = np.max(embeddings, axis=0)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Normalize
        result = result / np.linalg.norm(result)

        return result

    def get_performance_stats(self, model_name: str) -> Dict[str, float]:
        """
        Get performance statistics for a model

        Args:
            model_name: Name of model

        Returns:
            Dictionary of performance statistics
        """
        if model_name not in self.performance_metrics:
            return {}

        metrics = self.performance_metrics[model_name]
        if not metrics:
            return {}

        # Calculate averages
        avg_stats = {
            'avg_texts_per_second': np.mean([m.get('texts_per_second', 0) for m in metrics]),
            'avg_time_ms': np.mean([m.get('avg_time_ms', 0) for m in metrics]),
            'avg_memory_mb': np.mean([m.get('memory_mb', 0) for m in metrics]),
            'num_benchmarks': len(metrics)
        }

        return avg_stats

    def list_models(self) -> List[str]:
        """Get list of registered model names"""
        return list(self.model_registry.keys())

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.model_registry:
            del self.model_registry[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model '{model_name}' unloaded")

    def clear_all_models(self) -> None:
        """Unload all models from memory"""
        self.model_registry.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models cleared from memory")
