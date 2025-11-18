"""
Fine-tuner for domain-specific adaptation of embedding models
"""
import logging
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AdamW
)
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplet loss training"""

    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            triplets: List of (anchor, positive, negative) text tuples
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        # Tokenize all three texts
        anchor_enc = self.tokenizer(
            anchor,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        positive_enc = self.tokenizer(
            positive,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        negative_enc = self.tokenizer(
            negative,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'anchor': {k: v.squeeze(0) for k, v in anchor_enc.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_enc.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_enc.items()}
        }


class PairDataset(Dataset):
    """Dataset for pair-wise training"""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            pairs: List of (text1, text2) tuples
            labels: List of labels (1 for similar, 0 for dissimilar)
            tokenizer: Tokenizer
            max_length: Max sequence length
        """
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]

        enc1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        enc2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'text1': {k: v.squeeze(0) for k, v in enc1.items()},
            'text2': {k: v.squeeze(0) for k, v in enc2.items()},
            'label': torch.tensor(label, dtype=torch.float)
        }


class FineTuner:
    """
    Fine-tune models on domain-specific data
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: Optional[torch.device] = None
    ):
        """
        Initialize fine-tuner

        Args:
            base_model_name: Name of base model to fine-tune
            device: Device to use for training
        """
        self.base_model_name = base_model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"FineTuner initialized with {base_model_name} on {self.device}")

    def prepare_training_data(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Create training dataset from known good/bad matches

        Args:
            positive_pairs: List of (text1, text2) known to be similar
            negative_pairs: List of (text1, text2) known to be dissimilar

        Returns:
            List of triplets (anchor, positive, negative)
        """
        triplets = []

        # Create triplets from positive pairs
        for anchor, positive in positive_pairs:
            # Find a negative example
            if negative_pairs:
                # Sample a random negative
                negative = np.random.choice([n[1] for n in negative_pairs])
            else:
                # Use hard negative mining from positive pairs
                # (take a random different positive as negative)
                negatives = [p for p in positive_pairs if p != (anchor, positive)]
                if negatives:
                    negative = np.random.choice([n[1] for n in negatives])
                else:
                    continue

            triplets.append((anchor, positive, negative))

        logger.info(f"Prepared {len(triplets)} triplets for training")
        return triplets

    def fine_tune_for_similarity(
        self,
        training_triplets: List[Tuple[str, str, str]],
        validation_triplets: Optional[List[Tuple[str, str, str]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        margin: float = 0.5,
        output_dir: str = "models/custom_finetuned"
    ) -> str:
        """
        Fine-tune using contrastive learning with triplet loss

        Args:
            training_triplets: Training data triplets
            validation_triplets: Validation data triplets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            margin: Margin for triplet loss
            output_dir: Directory to save fine-tuned model

        Returns:
            Path to saved model
        """
        logger.info("Starting fine-tuning for similarity...")

        # Load model and tokenizer
        model = AutoModel.from_pretrained(self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        model = model.to(self.device)

        # Create dataset and dataloader
        train_dataset = TripletDataset(training_triplets, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Triplet loss
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

        # Training loop
        model.train()
        global_step = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch in progress_bar:
                # Move batch to device
                anchor_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['anchor'].items()
                }
                positive_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['positive'].items()
                }
                negative_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['negative'].items()
                }

                # Forward pass
                anchor_outputs = model(**anchor_inputs)
                positive_outputs = model(**positive_inputs)
                negative_outputs = model(**negative_inputs)

                # Mean pooling
                anchor_emb = self._mean_pooling(
                    anchor_outputs.last_hidden_state,
                    anchor_inputs['attention_mask']
                )
                positive_emb = self._mean_pooling(
                    positive_outputs.last_hidden_state,
                    positive_inputs['attention_mask']
                )
                negative_emb = self._mean_pooling(
                    negative_outputs.last_hidden_state,
                    negative_inputs['attention_mask']
                )

                # Calculate triplet loss
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            # Validation if provided
            if validation_triplets:
                val_loss = self._validate_similarity(
                    model,
                    tokenizer,
                    validation_triplets,
                    batch_size,
                    margin
                )
                logger.info(f"Validation loss: {val_loss:.4f}")

        # Save model
        import os
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Fine-tuned model saved to {output_dir}")
        return output_dir

    def fine_tune_for_classification(
        self,
        training_data: List[Tuple[str, int]],
        num_labels: int,
        validation_data: Optional[List[Tuple[str, int]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "models/classifier"
    ) -> str:
        """
        Fine-tune for research area classification

        Args:
            training_data: List of (text, label_id) tuples
            num_labels: Number of classification labels
            validation_data: Validation data
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Output directory

        Returns:
            Path to saved model
        """
        from transformers import AutoModelForSequenceClassification

        logger.info("Starting fine-tuning for classification...")

        # Load classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        model = model.to(self.device)

        # Prepare data
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]

        # Tokenize
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        # Create dataset
        class SimpleDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        train_dataset = SimpleDataset(encodings, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save model
        import os
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Classification model saved to {output_dir}")
        return output_dir

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _validate_similarity(
        self,
        model,
        tokenizer,
        validation_triplets: List[Tuple[str, str, str]],
        batch_size: int,
        margin: float
    ) -> float:
        """Validate on held-out triplets"""
        model.eval()

        val_dataset = TripletDataset(validation_triplets, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                anchor_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['anchor'].items()
                }
                positive_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['positive'].items()
                }
                negative_inputs = {
                    k: v.to(self.device)
                    for k, v in batch['negative'].items()
                }

                anchor_outputs = model(**anchor_inputs)
                positive_outputs = model(**positive_inputs)
                negative_outputs = model(**negative_inputs)

                anchor_emb = self._mean_pooling(
                    anchor_outputs.last_hidden_state,
                    anchor_inputs['attention_mask']
                )
                positive_emb = self._mean_pooling(
                    positive_outputs.last_hidden_state,
                    positive_inputs['attention_mask']
                )
                negative_emb = self._mean_pooling(
                    negative_outputs.last_hidden_state,
                    negative_inputs['attention_mask']
                )

                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()

        model.train()
        return total_loss / len(val_loader)
