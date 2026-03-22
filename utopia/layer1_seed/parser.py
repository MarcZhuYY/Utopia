"""Material parser for seed text classification and preprocessing.

This module handles the initial classification and validation of seed materials.
"""

from __future__ import annotations

import re
from typing import Optional

from utopia.core.models import MaterialType, SeedMaterial


# Keywords for material type classification
MATERIAL_TYPE_KEYWORDS = {
    MaterialType.POLICY: {
        "bill",
        "law",
        "regulation",
        "legislation",
        "congress",
        "parliament",
        "senate",
        "vote",
        "policy",
        "governance",
        "amendment",
        "act",
    },
    MaterialType.FINANCIAL: {
        "revenue",
        "earnings",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "quarterly",
        "annual",
        "market",
        "stock",
        "trading",
        "IPO",
        "bond",
        "interest rate",
        "inflation",
        "GDP",
        "unemployment",
        "Fed",
        "central bank",
        "fiscal",
        "recession",
        "bull",
        "bear",
        "dividend",
        "share",
        "portfolio",
    },
}


def classify_material(text: str) -> MaterialType:
    """Classify seed material type based on content analysis.

    Simple keyword-based classification with LLM fallback for ambiguous cases.

    Args:
        text: Raw text content

    Returns:
        MaterialType: Classified type (news/policy/financial/fiction)

    Raises:
        ValueError: If text is too short (<50 chars)
    """
    if not text or len(text.strip()) < 50:
        raise ValueError("Text too short (<50 chars). Please provide more content.")

    text_lower = text.lower()

    # Count keyword matches for each type
    scores: dict[MaterialType, int] = {MaterialType.NEWS: 0, MaterialType.POLICY: 0, MaterialType.FINANCIAL: 0}

    for mtype, keywords in MATERIAL_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                scores[mtype] += 1

    # Check for fiction indicators
    fiction_indicators = ['"', "chapter", "narrative", "character", "story", "novel"]
    fiction_count = sum(1 for w in fiction_indicators if w in text_lower)

    if fiction_count >= 2:
        return MaterialType.FICTION

    # Return type with highest score, default to NEWS
    max_score = max(scores.values())
    if max_score > 0:
        for mtype, score in scores.items():
            if score == max_score:
                return mtype

    return MaterialType.NEWS


class MaterialParser:
    """Parser for seed material preprocessing and validation.

    Handles text chunking for long materials and basic cleaning.
    """

    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 10000
    MAX_ENTITIES = 100

    def __init__(self, max_text_length: int = MAX_TEXT_LENGTH, max_entities: int = MAX_ENTITIES):
        """Initialize parser with limits.

        Args:
            max_text_length: Maximum text length before chunking
            max_entities: Maximum entities to retain (keep highest influence)
        """
        self.max_text_length = max_text_length
        self.max_entities = max_entities

    def parse(self, raw_text: str) -> SeedMaterial:
        """Parse raw text into structured SeedMaterial.

        Args:
            raw_text: Raw input text

        Returns:
            SeedMaterial: Parsed and classified material

        Raises:
            ValueError: If text is invalid
        """
        # Clean text
        cleaned_text = self._clean_text(raw_text)

        # Validate length
        if len(cleaned_text) < self.MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short: {len(cleaned_text)} chars (min: {self.MIN_TEXT_LENGTH})")

        # Classify material type
        material_type = classify_material(cleaned_text)

        # Check if chunking needed
        if len(cleaned_text) > self.max_text_length:
            # For MVP, just truncate (can be enhanced with semantic chunking)
            cleaned_text = cleaned_text[: self.max_text_length]

        return SeedMaterial(raw_text=cleaned_text, material_type=material_type)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)
        # Remove extra newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 5000) -> list[str]:
        """Split long text into chunks for processing.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters

        Returns:
            list[str]: List of text chunks
        """
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
