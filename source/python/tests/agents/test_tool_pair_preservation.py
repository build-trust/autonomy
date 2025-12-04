"""
Tests for tool_use/tool_result pair preservation in memory trimming and context summarization.

This test file validates the fix for the bug where:
1. Memory trimming (FIFO) could remove tool_use messages while leaving orphaned tool_results
2. Context summarization could set a verbatim boundary that splits tool pairs

Both issues cause Claude API errors: "Expected toolResult blocks at messages.X.content for the following Ids: tooluse_XXX"
"""

import pytest
from autonomy.memory.memory import (
  Memory,
  get_tool_use_ids,
  get_tool_result_id,
  find_tool_pair_boundary,
  find_safe_trim_count,
  validate_tool_pairing,
)


# =============================================================================
# SECTION 1: Unit Tests for Tool Pair Detection Utilities
# =============================================================================


class TestGetToolUseIds:
  """Tests for get_tool_use_ids function"""

  def test_assistant_message_with_single_tool_call(self):
    """Extract single tool call ID from assistant message"""
    message = {
      "role": "assistant",
      "content": "",
      "tool_calls": [{"id": "tool_123", "name": "search", "arguments": "{}"}],
    }
    assert get_tool_use_ids(message) == ["tool_123"]

  def test_assistant_message_with_multiple_tool_calls(self):
    """Extract multiple tool call IDs from assistant message"""
    message = {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "tool_1", "name": "search", "arguments": "{}"},
        {"id": "tool_2", "name": "calculator", "arguments": "{}"},
        {"id": "tool_3", "name": "weather", "arguments": "{}"},
      ],
    }
    assert get_tool_use_ids(message) == ["tool_1", "tool_2", "tool_3"]

  def test_assistant_message_without_tool_calls(self):
    """No IDs from regular assistant message"""
    message = {"role": "assistant", "content": "Hello!"}
    assert get_tool_use_ids(message) == []

  def test_user_message(self):
    """No IDs from user message"""
    message = {"role": "user", "content": "Hello!"}
    assert get_tool_use_ids(message) == []

  def test_tool_message(self):
    """No IDs from tool result message"""
    message = {"role": "tool", "content": "Result", "tool_call_id": "tool_123"}
    assert get_tool_use_ids(message) == []

  def test_empty_tool_calls_list(self):
    """Empty list when tool_calls is empty"""
    message = {"role": "assistant", "content": "", "tool_calls": []}
    assert get_tool_use_ids(message) == []

  def test_tool_call_without_id(self):
    """Skip tool calls without ID"""
    message = {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "tool_1", "name": "search"},
        {"name": "calculator"},  # No ID
        {"id": "tool_3", "name": "weather"},
      ],
    }
    assert get_tool_use_ids(message) == ["tool_1", "tool_3"]


class TestGetToolResultId:
  """Tests for get_tool_result_id function"""

  def test_tool_message_with_id(self):
    """Extract tool_call_id from tool result message"""
    message = {"role": "tool", "content": "Result", "tool_call_id": "tool_123"}
    assert get_tool_result_id(message) == "tool_123"

  def test_tool_message_without_id(self):
    """None when tool message has no tool_call_id"""
    message = {"role": "tool", "content": "Result"}
    assert get_tool_result_id(message) is None

  def test_user_message(self):
    """None for user message"""
    message = {"role": "user", "content": "Hello!"}
    assert get_tool_result_id(message) is None

  def test_assistant_message(self):
    """None for assistant message"""
    message = {"role": "assistant", "content": "Hello!"}
    assert get_tool_result_id(message) is None


class TestFindToolPairBoundary:
  """Tests for find_tool_pair_boundary function"""

  def test_no_tool_calls(self):
    """No adjustment needed when no tool calls"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi"},
      {"role": "user", "content": "Bye"},
    ]
    assert find_tool_pair_boundary(messages, 1) == 1
    assert find_tool_pair_boundary(messages, 2) == 2

  def test_boundary_does_not_split_pair(self):
    """No adjustment when boundary doesn't split a pair"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
      {"role": "user", "content": "Thanks"},
    ]
    # Boundary at 3 is safe - pair is at indices 1-2
    assert find_tool_pair_boundary(messages, 3) == 3

  def test_boundary_splits_pair_adjusted_backwards(self):
    """Adjust backwards when boundary splits a tool pair"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
      {"role": "user", "content": "Thanks"},
    ]
    # Boundary at 2 would include tool result but exclude tool_use
    # Should adjust to 1 to include the tool_use
    assert find_tool_pair_boundary(messages, 2) == 1

  def test_multiple_pairs_boundary_at_middle(self):
    """Handle multiple tool pairs with boundary in middle"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "a"}]},
      {"role": "tool", "content": "R1", "tool_call_id": "tool_1"},
      {"role": "assistant", "tool_calls": [{"id": "tool_2", "name": "b"}]},
      {"role": "tool", "content": "R2", "tool_call_id": "tool_2"},
      {"role": "user", "content": "Thanks"},
    ]
    # Boundary at 4 would split second pair (tool_use at 3, result at 4)
    # Should adjust to 3
    assert find_tool_pair_boundary(messages, 4) == 3
    # Boundary at 5 is safe
    assert find_tool_pair_boundary(messages, 5) == 5

  def test_boundary_at_start(self):
    """Boundary at 0 stays at 0"""
    messages = [
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
    ]
    assert find_tool_pair_boundary(messages, 0) == 0

  def test_boundary_beyond_messages(self):
    """Boundary beyond messages stays unchanged"""
    messages = [{"role": "user", "content": "Hello"}]
    assert find_tool_pair_boundary(messages, 10) == 10

  def test_assistant_multiple_tools_single_message(self):
    """Handle assistant message with multiple tool calls"""
    messages = [
      {"role": "user", "content": "Hello"},
      {
        "role": "assistant",
        "tool_calls": [
          {"id": "tool_1", "name": "a"},
          {"id": "tool_2", "name": "b"},
        ],
      },
      {"role": "tool", "content": "R1", "tool_call_id": "tool_1"},
      {"role": "tool", "content": "R2", "tool_call_id": "tool_2"},
      {"role": "user", "content": "Thanks"},
    ]
    # Boundary at 2 would include first result but exclude tool_use
    assert find_tool_pair_boundary(messages, 2) == 1
    # Boundary at 3 would include both results but exclude tool_use
    assert find_tool_pair_boundary(messages, 3) == 1
    # Boundary at 4 is safe
    assert find_tool_pair_boundary(messages, 4) == 4


class TestFindSafeTrimCount:
  """Tests for find_safe_trim_count function"""

  def test_no_messages(self):
    """Empty messages returns 0"""
    assert find_safe_trim_count([], 5) == 0

  def test_zero_trim_count(self):
    """Zero requested returns 0"""
    messages = [{"role": "user", "content": "Hello"}]
    assert find_safe_trim_count(messages, 0) == 0

  def test_no_tool_calls(self):
    """Full trim when no tool calls"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi"},
      {"role": "user", "content": "Bye"},
    ]
    assert find_safe_trim_count(messages, 2) == 2
    assert find_safe_trim_count(messages, 3) == 3

  def test_trim_before_tool_pair_safe(self):
    """Safe to trim messages before a tool pair"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi"},
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
    ]
    # Can safely trim first 2 messages
    assert find_safe_trim_count(messages, 2) == 2

  def test_trim_would_orphan_result(self):
    """Reduce trim count when it would orphan tool result"""
    messages = [
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
      {"role": "user", "content": "Thanks"},
    ]
    # Trimming 1 would remove tool_use but leave result - not safe
    # Should reduce to 0
    assert find_safe_trim_count(messages, 1) == 0
    # Trimming 2 would remove both - safe
    assert find_safe_trim_count(messages, 2) == 2

  def test_multiple_pairs_complex(self):
    """Handle complex scenario with multiple pairs"""
    messages = [
      {"role": "user", "content": "Q1"},  # 0
      {"role": "assistant", "tool_calls": [{"id": "t1", "name": "a"}]},  # 1
      {"role": "tool", "content": "R1", "tool_call_id": "t1"},  # 2
      {"role": "user", "content": "Q2"},  # 3
      {"role": "assistant", "tool_calls": [{"id": "t2", "name": "b"}]},  # 4
      {"role": "tool", "content": "R2", "tool_call_id": "t2"},  # 5
    ]
    # Trim 1: safe (just user message)
    assert find_safe_trim_count(messages, 1) == 1
    # Trim 2: would orphan t1 result at index 2 - reduce to 1
    assert find_safe_trim_count(messages, 2) == 1
    # Trim 3: would include t1 pair but orphan nothing - safe
    assert find_safe_trim_count(messages, 3) == 3
    # Trim 4: safe (pair at 1-2 fully trimmed, index 3 is user)
    assert find_safe_trim_count(messages, 4) == 4
    # Trim 5: would orphan t2 result - reduce to 4
    assert find_safe_trim_count(messages, 5) == 4


class TestValidateToolPairing:
  """Tests for validate_tool_pairing function"""

  def test_no_tools_valid(self):
    """Valid when no tool calls or results"""
    messages = [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi"},
    ]
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid
    assert error is None

  def test_paired_tools_valid(self):
    """Valid when all tools are properly paired"""
    messages = [
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
    ]
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid
    assert error is None

  def test_orphaned_result_invalid(self):
    """Invalid when tool result has no matching tool_use"""
    messages = [
      {"role": "tool", "content": "Result", "tool_call_id": "tool_1"},
      {"role": "assistant", "content": "Based on results..."},
    ]
    is_valid, error = validate_tool_pairing(messages)
    assert not is_valid
    assert "tool_1" in error

  def test_multiple_paired_valid(self):
    """Valid with multiple properly paired tools"""
    messages = [
      {
        "role": "assistant",
        "tool_calls": [
          {"id": "t1", "name": "a"},
          {"id": "t2", "name": "b"},
        ],
      },
      {"role": "tool", "content": "R1", "tool_call_id": "t1"},
      {"role": "tool", "content": "R2", "tool_call_id": "t2"},
    ]
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid
    assert error is None

  def test_pending_tool_use_still_valid(self):
    """Tool use without result is valid (may be pending)"""
    messages = [
      {"role": "assistant", "tool_calls": [{"id": "tool_1", "name": "search"}]},
      # No result yet - this is okay
    ]
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid
    assert error is None


# =============================================================================
# SECTION 2: Integration Tests for Memory Trimming
# =============================================================================


class TestMemoryTrimmingToolPairs:
  """Tests for tool pair preservation in Memory._trim_short_term_memory"""

  @pytest.fixture
  def memory_with_low_limit(self):
    """Create memory with low message limit for testing trimming"""
    return Memory(max_messages_in_short_term_memory=3)

  @pytest.mark.asyncio
  async def test_trim_preserves_tool_pairs(self, memory_with_low_limit):
    """Trimming should preserve tool_use/tool_result pairs"""
    scope, conv = "test", "conv1"

    # Add messages that include a tool pair
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "user", "content": "Search for X"},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "assistant", "tool_calls": [{"id": "t1", "name": "search"}]},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "tool", "content": "Results", "tool_call_id": "t1"},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "assistant", "content": "Here are the results"},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "user", "content": "Thanks"},
    )

    # Get messages after trimming
    messages = await memory_with_low_limit.get_messages_only(scope, conv)

    # Validate tool pairing is preserved
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid, f"Tool pairing broken after trimming: {error}"

  @pytest.mark.asyncio
  async def test_trim_with_multiple_tool_pairs(self, memory_with_low_limit):
    """Trimming with multiple tool pairs should keep them intact"""
    scope, conv = "test", "conv2"

    # Add first tool pair
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "assistant", "tool_calls": [{"id": "t1", "name": "a"}]},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "tool", "content": "R1", "tool_call_id": "t1"},
    )

    # Add second tool pair
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "assistant", "tool_calls": [{"id": "t2", "name": "b"}]},
    )
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "tool", "content": "R2", "tool_call_id": "t2"},
    )

    # Add more messages to trigger trimming
    await memory_with_low_limit.add_message(
      scope,
      conv,
      {"role": "user", "content": "Question"},
    )

    # Get messages after trimming
    messages = await memory_with_low_limit.get_messages_only(scope, conv)

    # Validate tool pairing is preserved
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid, f"Tool pairing broken after trimming: {error}"

  @pytest.mark.asyncio
  async def test_trim_long_conversation_with_tools(self):
    """Simulate long conversation with many tool uses"""
    memory = Memory(max_messages_in_short_term_memory=5)
    scope, conv = "test", "long_conv"

    # Add many messages with tool pairs
    for i in range(10):
      await memory.add_message(
        scope,
        conv,
        {"role": "user", "content": f"Question {i}"},
      )
      await memory.add_message(
        scope,
        conv,
        {"role": "assistant", "tool_calls": [{"id": f"t{i}", "name": "search"}]},
      )
      await memory.add_message(
        scope,
        conv,
        {"role": "tool", "content": f"Result {i}", "tool_call_id": f"t{i}"},
      )
      await memory.add_message(
        scope,
        conv,
        {"role": "assistant", "content": f"Answer {i}"},
      )

    # Get messages after trimming
    messages = await memory.get_messages_only(scope, conv)

    # Should be trimmed to around the limit
    assert len(messages) <= 10  # Allow some flexibility for pair preservation

    # Validate tool pairing is preserved
    is_valid, error = validate_tool_pairing(messages)
    assert is_valid, f"Tool pairing broken in long conversation: {error}"


# =============================================================================
# SECTION 3: Edge Cases
# =============================================================================


class TestEdgeCases:
  """Edge case tests for tool pair preservation"""

  def test_tool_result_without_tool_use_in_context(self):
    """
    Edge case: tool result exists but tool_use was summarized away.
    This is the bug scenario we're fixing.
    """
    # This represents a broken state that could occur without the fix
    messages = [
      # tool_use was summarized, only result remains
      {"role": "tool", "content": "Search results", "tool_call_id": "orphan_123"},
      {"role": "assistant", "content": "Based on the search..."},
      {"role": "user", "content": "Thanks!"},
    ]

    is_valid, error = validate_tool_pairing(messages)
    assert not is_valid
    assert "orphan_123" in error

  def test_boundary_adjustment_needed(self):
    """Test scenario where boundary must be adjusted"""
    messages = [
      {"role": "user", "content": "Hello"},  # 0 - could be summarized
      {"role": "assistant", "tool_calls": [{"id": "t1", "name": "s"}]},  # 1 - could be summarized
      {"role": "tool", "content": "R", "tool_call_id": "t1"},  # 2 - verbatim window starts here?
      {"role": "assistant", "content": "Done"},  # 3
    ]

    # If verbatim_start=2, we'd have orphaned tool result
    # find_tool_pair_boundary should adjust to 1
    boundary = find_tool_pair_boundary(messages, 2)
    assert boundary == 1

    # Verify the adjusted window is valid
    verbatim_messages = messages[boundary:]
    is_valid, error = validate_tool_pairing(verbatim_messages)
    assert is_valid, f"Adjusted boundary still invalid: {error}"

  def test_nested_tool_calls_same_assistant_message(self):
    """Handle assistant message with multiple tool calls"""
    messages = [
      {"role": "user", "content": "Do both tasks"},
      {
        "role": "assistant",
        "tool_calls": [
          {"id": "t1", "name": "task1"},
          {"id": "t2", "name": "task2"},
        ],
      },
      {"role": "tool", "content": "R1", "tool_call_id": "t1"},
      {"role": "tool", "content": "R2", "tool_call_id": "t2"},
      {"role": "assistant", "content": "Both done"},
    ]

    # Boundary at 2 would orphan t1 result but also need t2
    # Should go back to 1 (the assistant message with both tool calls)
    boundary = find_tool_pair_boundary(messages, 2)
    assert boundary == 1

    # Boundary at 3 would orphan t2 result
    boundary = find_tool_pair_boundary(messages, 3)
    assert boundary == 1

    # Boundary at 4 is safe
    boundary = find_tool_pair_boundary(messages, 4)
    assert boundary == 4
