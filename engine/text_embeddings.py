"""Optional frozen text embeddings for S03; no model downloads or loading.

Inject a provider into TextEmbeddingCache and reuse that cache across candidates.
The adapter accepts an already-loaded SentenceTransformer; the caller owns its
installation and offline model assets. S02 and the baseline never import here.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Protocol

import torch


TEXT_DIMENSION = 384


class EmbeddingUnavailable(RuntimeError):
    """An encoder cannot provide a vector; graph masks will expose absence."""


class EmbeddingProvider(Protocol):
    @property
    def encoder_version(self) -> str:
        """Stable model/revision/preprocessing identity, not just a display name."""
        ...

    def encode(self, text: str) -> Sequence[float] | torch.Tensor | None:
        """Encode literal text or return None/raise EmbeddingUnavailable."""
        ...


class TextEmbeddingCache:
    """Own CPU float32 vectors keyed by (encoder version, SHA-256 of content).

    Missing/zero-norm results are cached too. Clear the cache to retry a transient
    failure. Malformed vectors raise ValueError, rather than masquerading as an
    unavailable model. Every returned tensor is detached and independently owned.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._cache: dict[tuple[str, str], torch.Tensor | None] = {}

    @property
    def encoder_version(self) -> str:
        version = self.provider.encoder_version
        if not isinstance(version, str) or not version:
            raise ValueError("embedding provider must declare a nonempty encoder_version")
        return version

    def clear(self) -> None:
        self._cache.clear()

    def get(self, text: str) -> torch.Tensor | None:
        # surrogatepass preserves even escaped, unpaired Unicode surrogates.
        digest = sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
        key = (self.encoder_version, digest)
        if key not in self._cache:
            try:
                with torch.inference_mode():
                    raw = self.provider.encode(text)
            except EmbeddingUnavailable:
                raw = None
            vector = None
            if raw is not None:
                vector = torch.as_tensor(raw, dtype=torch.float32, device="cpu").detach().clone()
                if vector.shape != (TEXT_DIMENSION,):
                    raise ValueError(f"text embedding must have shape ({TEXT_DIMENSION},)")
                if not torch.isfinite(vector).all().item():
                    raise ValueError("text embedding must contain finite float32 values")
                if not torch.any(vector != 0).item():
                    vector = None  # No defined cosine direction.
            self._cache[key] = vector
        cached = self._cache[key]
        return cached.clone() if cached is not None else None


class FrozenSentenceTransformerProvider:
    """Adapt an already-loaded 384-dimensional sentence transformer.

    For the proposed baseline, load sentence-transformers/all-MiniLM-L6-v2 from
    local assets before constructing this adapter. It never downloads anything.
    """

    def __init__(self, model: Any, *, encoder_version: str) -> None:
        if not isinstance(encoder_version, str) or not encoder_version:
            raise ValueError("encoder_version must identify the model and revision")
        if model.get_sentence_embedding_dimension() != TEXT_DIMENSION:
            raise ValueError(f"encoder dimension must be {TEXT_DIMENSION}")
        self.model = model
        self._encoder_version = encoder_version
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def encoder_version(self) -> str:
        return self._encoder_version

    def encode(self, text: str) -> torch.Tensor:
        self.model.eval()
        try:
            with torch.inference_mode():
                return self.model.encode(
                    [text], convert_to_tensor=True, show_progress_bar=False,
                    normalize_embeddings=False,
                )[0]
        except (OSError, RuntimeError) as exc:
            raise EmbeddingUnavailable("frozen text encoder inference unavailable") from exc
