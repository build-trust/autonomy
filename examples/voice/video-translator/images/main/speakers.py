"""Match and manage speaker labels for multi-chunk transcription.

This module provides functions to:
- Match speakers across transcript chunks using LLM analysis
- Prefix speaker labels with chunk numbers
- Apply speaker label mappings to transcripts
"""

import json
import re

from typing import Optional
from autonomy import Model


SPEAKER_MATCHER_INSTRUCTIONS = """
You are an expert at analyzing transcripts to identify and match speakers across different segments.

You will receive a transcript that was transcribed in multiple chunks. Each chunk has its own speaker labels
(e.g., "Speaker 1A", "Speaker 1B" for chunk 1, "Speaker 2A", "Speaker 2B" for chunk 2).

Your job is to analyze the transcript and return ONLY a JSON mapping of speaker labels.
DO NOT rewrite the transcript - just provide the mapping.

CLUES TO LOOK FOR:
- Self-introductions: "I'm Maria", "My name is Dr. Lopez"
- Being addressed by name: "Thank you, Maria", "Dr. Lopez, what do you think?"
- Role indicators: "As the host...", "In my research...", "As your reporter..."
- Speaking patterns: Who asks questions (host) vs who answers (guest)
- Topic expertise: Same person discussing same topic across chunks

OUTPUT FORMAT (JSON only, no other text):
{
  "mapping": {
    "Speaker 1A": "Maria (Host)",
    "Speaker 1B": "Dr. Lopez (Guest)",
    "Speaker 2A": "Maria (Host)",
    "Speaker 2B": "Reporter",
    "Speaker 2C": "Dr. Lopez (Guest)"
  },
  "confidence": "high",
  "notes": "Maria identified as host from introduction. Dr. Lopez mentioned by name in chunk 2."
}

RULES:
- Return ONLY valid JSON, no other text before or after
- Map each chunk-prefixed label (e.g., "Speaker 1A") to a unified name
- Use real names when mentioned in the transcript, otherwise use roles (Host, Guest, Reporter, etc.)
- Same person across chunks should map to the exact same unified name
- If you cannot determine a match with confidence, keep the original label as the value
- "confidence" should be "high", "medium", or "low"
- "notes" should briefly explain your reasoning
"""


def apply_speaker_mapping(transcript: str, mapping: dict) -> str:
    """Apply a speaker label mapping to a transcript.

    Replaces chunk-prefixed labels (e.g., "Speaker 1A:") with unified names
    (e.g., "Maria (Host):").

    Args:
      transcript: Original transcript with chunk-prefixed labels
      mapping: Dict mapping old labels to new labels

    Returns:
      Transcript with unified speaker labels
    """
    result = transcript

    sorted_labels = sorted(mapping.keys(), key=len, reverse=True)

    for old_label in sorted_labels:
        new_label = mapping[old_label]
        # Replace "Speaker 1A:" with "Maria (Host):" etc.
        result = re.sub(rf"\b{re.escape(old_label)}:", f"{new_label}:", result)

    return result


def extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON object from model response, handling various formats.

    Handles cases where the model returns:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after
    - Just the mapping without outer braces

    Args:
      text: Raw model response text

    Returns:
      Parsed JSON dict, or None if parsing fails
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(
        r'\{[^{}]*"mapping"\s*:\s*\{[^{}]*\}[^{}]*\}', text, re.DOTALL
    )
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    mapping_match = re.search(r'"mapping"\s*:\s*(\{[^{}]*\})', text, re.DOTALL)
    if mapping_match:
        try:
            mapping = json.loads(mapping_match.group(1))
            return {"mapping": mapping}
        except json.JSONDecodeError:
            pass

    speaker_patterns = re.findall(r'"(Speaker \d+[A-Z])"\s*:\s*"([^"]+)"', text)
    if speaker_patterns:
        return {"mapping": dict(speaker_patterns)}

    return None


async def match_speakers_across_chunks(transcript: str) -> str:
    """Unify speaker labels across transcript chunks using LLM analysis.

    Uses a two-step approach for efficiency:
    1. Ask the LLM to return ONLY a JSON mapping of speaker labels (fast, small output)
    2. Apply the mapping programmatically (instant)

    This avoids timeout issues caused by asking the LLM to rewrite the entire transcript.

    Args:
      transcript: Multi-chunk transcript with prefixed speaker labels (e.g., Speaker 1A, 2A)

    Returns:
      Unified transcript with consistent speaker labels, or original transcript if matching fails
    """
    print(f"Analyzing {len(transcript)} characters for speaker mapping...")

    model = Model("claude-sonnet-4-5")
    messages = [
        {"role": "system", "content": SPEAKER_MATCHER_INSTRUCTIONS},
        {
            "role": "user",
            "content": f"Analyze this transcript and return a JSON speaker mapping:\n\n{transcript}",
        },
    ]

    try:
        response = await model.complete_chat(messages, stream=False)

        if hasattr(response, "choices") and len(response.choices) > 0:
            result = response.choices[0].message.content.strip()
        else:
            print("No response from model")
            return transcript

        mapping_data = extract_json_from_response(result)

        if mapping_data is None:
            print("Failed to parse speaker mapping JSON from response")
            print(f"Raw response (first 500 chars): {result[:500]}")
            return transcript
        mapping = mapping_data.get("mapping", {})
        confidence = mapping_data.get("confidence", "unknown")
        notes = mapping_data.get("notes", "")

        print(
            f"Speaker mapping received (confidence: {confidence}): {len(mapping)} labels mapped"
        )
        if notes:
            print(f"  Notes: {notes}")

        if not mapping:
            print("Empty mapping received, returning original transcript")
            return transcript

        unified_transcript = apply_speaker_mapping(transcript, mapping)
        unified_transcript = cleanup_transcript_artifacts(unified_transcript)

        return unified_transcript

    except Exception as e:
        print(f"Error in speaker matching: {e}")
        return transcript


def prefix_speakers_with_chunk(transcript: str, chunk_num: int) -> str:
    """Prefix speaker labels with chunk number to differentiate across chunks.

    Transforms:
    - "Speaker A:" to "Speaker 1A:" for chunk 1
    - "Speaker 1:" to "Speaker 1_1:" (numeric labels from failed diarization)
    - "Unknown Speaker:" to "Unknown Speaker 1:" for chunk 1

    Args:
      transcript: Transcript text with speaker labels
      chunk_num: The chunk number (1-based)

    Returns:
      Transcript with chunk-prefixed speaker labels
    """
    result = transcript

    # Match "Speaker X:" where X is a single letter (most common case)
    def replace_speaker_letter(match):
        letter = match.group(1)
        return f"Speaker {chunk_num}{letter}:"

    result = re.sub(r"Speaker ([A-Z]):", replace_speaker_letter, result)

    # Match "Speaker N:" where N is a number (from Whisper fallback or edge cases)
    def replace_speaker_number(match):
        number = match.group(1)
        return f"Speaker {chunk_num}_{number}:"

    result = re.sub(r"Speaker (\d+):", replace_speaker_number, result)

    # Match "Unknown Speaker:" and prefix with chunk number
    result = re.sub(r"Unknown Speaker:", f"Unknown Speaker {chunk_num}:", result)

    return result


def cleanup_transcript_artifacts(transcript: str) -> str:
    """Clean up any remaining artifacts from transcription and speaker matching.

    Performs final normalization to catch edge cases that might slip through:
    - Stray @: labels that weren't caught earlier
    - Orphan "Speaker N:" labels without chunk prefix
    - Multiple consecutive spaces
    - Empty speaker segments

    Args:
      transcript: Transcript text to clean up

    Returns:
      Cleaned transcript text
    """
    result = transcript

    # Replace any remaining @: with Unknown Speaker:
    result = re.sub(r"\b@:", "Unknown Speaker:", result)

    # Replace bare @ at line start followed by text
    result = re.sub(r"^@\s+", "Unknown Speaker: ", result, flags=re.MULTILINE)

    # Clean up multiple consecutive spaces
    result = re.sub(r"  +", " ", result)

    # Remove empty speaker lines (just "Speaker X:" with no content)
    result = re.sub(r"^[^:\n]+:\s*$\n?", "", result, flags=re.MULTILINE)

    # Clean up multiple consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
