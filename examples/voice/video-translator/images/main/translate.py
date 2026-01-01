"""Translate transcribed text to English using LLM agents.

This module provides functions to translate transcribed text to English
using the translator agent, with chunked processing for long transcripts.
"""

from autonomy import Node, Agent, Model
from typing import Optional

# Maximum characters per translation chunk to avoid output token limits
# ~15,000 chars input typically produces ~12,000 chars output, staying under token limits
MAX_CHARS_PER_CHUNK = 15000


TRANSLATOR_INSTRUCTIONS = """
You are a professional translator. You will receive transcribed text that may be
in any language. The transcription includes speaker labels (e.g., "Speaker A:", "Speaker 1A:", "Maria:", "Unknown Speaker:", etc.).

Your job is to:
1. Identify the source language
2. Translate the text accurately to English
3. PRESERVE all existing speaker labels from the transcription exactly as they appear

Output format:
---
Source Language: [detected language]

Translation:
[speaker label]: [translated text]
---

CRITICAL RULES:
- PRESERVE existing speaker labels EXACTLY as they appear in the input
- DO NOT modify, rename, or add any speaker labels
- DO NOT add "Speaker 1:", "Speaker 2:", etc. - the input already has proper labels
- Keep each speaker's dialogue on separate lines with their original label
- Maintain natural, fluent English while preserving the original meaning
- Do not add any commentary or notes - just provide the translation
"""


_agent: Optional[Agent] = None


def is_ready() -> bool:
    """Check if the translator agent is initialized.

    Returns:
      True if the translator agent is ready to use
    """
    return _agent is not None


def get_agent() -> Agent:
    """Get the translator agent singleton.

    Returns:
      The translator agent instance

    Raises:
      RuntimeError: If agent not initialized
    """
    if _agent is None:
        raise RuntimeError("Translator agent not initialized. Call initialize() first.")
    return _agent


async def initialize(node: Node) -> bool:
    """Initialize the translator agent.

    Args:
      node: Autonomy node instance

    Returns:
      True if initialized successfully
    """
    global _agent

    _agent = await Agent.start(
        node=node,
        name="translator",
        instructions=TRANSLATOR_INSTRUCTIONS,
        model=Model("claude-sonnet-4-5"),
    )

    print("Translator agent initialized")
    return True


def split_transcript_into_chunks(
    text: str, max_chars: int = MAX_CHARS_PER_CHUNK
) -> list[str]:
    """Split transcript into chunks at speaker boundaries.

    Splits on double newlines (speaker boundaries) to keep each speaker's
    dialogue intact within a chunk.

    Args:
      text: Full transcript text
      max_chars: Maximum characters per chunk

    Returns:
      List of transcript chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = []
    current_length = 0

    segments = text.split("\n\n")

    for segment in segments:
        segment_length = len(segment) + 2  # +2 for the newlines we'll add back

        # If single segment exceeds max, we have to include it anyway
        if segment_length > max_chars and not current_chunk:
            chunks.append(segment)
            continue

        # If adding this segment exceeds max, start new chunk
        if current_length + segment_length > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(segment)
        current_length += segment_length

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


async def translate_chunk(text: str, chunk_num: int = 1, total_chunks: int = 1) -> str:
    """Translate a single chunk of transcript.

    Args:
      text: Text chunk to translate
      chunk_num: Current chunk number (for logging)
      total_chunks: Total number of chunks (for logging)

    Returns:
      Translated text
    """
    agent = get_agent()

    base_timeout = 180
    length_based_timeout = len(text) / 100
    timeout = int(min(600, max(base_timeout, base_timeout + length_based_timeout)))

    if total_chunks > 1:
        print(
            f"Translating chunk {chunk_num}/{total_chunks} ({len(text)} chars, {timeout}s timeout)..."
        )
    else:
        print(f"Translating {len(text)} characters with {timeout}s timeout...")

    response = await agent.send(
        f"Please translate the following transcription to English:\n\n{text}",
        timeout=timeout,
    )

    return response[-1].content.text


def clean_translation_output(text: str) -> str:
    """Remove translation format headers from output.

    Removes "---", "Source Language:", "Translation:" markers that the
    translator adds.

    Args:
      text: Raw translation output

    Returns:
      Cleaned translation text
    """
    import re

    # Remove leading/trailing --- delimiters
    cleaned = re.sub(r"^---\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*---$", "", cleaned, flags=re.MULTILINE)

    # Remove "Source Language: ..." line
    cleaned = re.sub(
        r"^Source Language:\s*\w+\s*\n*",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Remove "Translation:" label if on its own line
    cleaned = re.sub(
        r"^Translation:\s*\n*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    )

    return cleaned.strip()


async def translate_to_english(text: str, progress_callback=None) -> str:
    """Translate transcribed text to English using the agent.

    For long transcripts, splits into chunks to avoid output token limits.
    Each chunk is translated separately and results are combined.

    Args:
      text: The text to translate
      progress_callback: Optional async callback(message, percent) for progress updates

    Returns:
      Translated text in English

    Raises:
      RuntimeError: If translator agent is not initialized
    """
    get_agent()
    chunks = split_transcript_into_chunks(text)
    total_chunks = len(chunks)

    if total_chunks == 1:
        result = await translate_chunk(text)
        return clean_translation_output(result)

    print(
        f"Transcript too long ({len(text)} chars), splitting into {total_chunks} chunks for translation..."
    )

    translated_chunks = []
    for i, chunk in enumerate(chunks, 1):
        if progress_callback:
            progress = 85 + int((i - 1) / total_chunks * 14)
            await progress_callback(
                f"Translating to English (part {i}/{total_chunks})...", progress
            )

        translated = await translate_chunk(chunk, i, total_chunks)
        cleaned = clean_translation_output(translated)
        translated_chunks.append(cleaned)

    combined = "\n\n".join(translated_chunks)
    print(
        f"Translation complete: {len(combined)} characters from {total_chunks} chunks"
    )

    return combined
