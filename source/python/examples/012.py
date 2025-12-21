"""
Example 012: Agent Persistence Test Over Many Iterations

This example tests whether an agent can maintain focus, consistency, and
instruction adherence over a large number of iterations (default: 100).

The agent processes a series of log files sequentially, extracting security
incidents and maintaining a master tracking document. This validates:

1. Completion - processes all N files without getting lost
2. Accuracy - extracts the correct incidents (no false positives/negatives)
3. Consistency - maintains report format from file 1 to file N
4. No hallucinations - all reported incidents are real
5. Progress tracking - properly tracks file numbering and checkpoints
6. Memory - checkpoint summaries accurately reflect previous work

This is designed to uncover:
- Context window limitations
- Instruction drift over many iterations
- Agent fatigue (skipping validation steps)
- Hallucinations (fabricating data to complete tasks faster)
- Off-course behavior (deviating from original instructions)

Configuration:
  Set NUM_ITERATIONS environment variable to change iteration count (default: 100)
  Set SEED environment variable for deterministic test data generation

  Example: NUM_ITERATIONS=50 SEED=42 uv run --active examples/012.py

RECOMMENDED: Run with transcript mode and in-memory database:
  NUM_ITERATIONS=100 \
  AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  AUTONOMY_TRANSCRIPTS_DIR=/tmp/transcripts \
  uv run --active examples/012.py

With AWS Bedrock:
  NUM_ITERATIONS=100 \
  AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  AUTONOMY_TRANSCRIPTS_DIR=/tmp/transcripts \
  uv run --active examples/012.py
"""

from autonomy import Agent, Model, Node, info
from autonomy.tools.filesystem import FilesystemTools
import os
import tempfile
import shutil
import random
import json
from datetime import datetime, timedelta
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of log files to generate (configurable via environment variable)
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "100"))

# Checkpoint interval (agent should summarize progress every N files)
CHECKPOINT_INTERVAL = max(10, NUM_ITERATIONS // 10)  # At least 10, or 10% of total

# Timeout for agent task (30 minutes to handle large iteration counts)
TIMEOUT = 1800  # 30 minutes

# Random seed for deterministic test data generation
SEED = os.getenv("SEED")
if SEED is not None:
  try:
    random.seed(int(SEED))
    info(f"✓ Using deterministic seed: {SEED}")
  except ValueError:
    random.seed(SEED)
    info(f"✓ Using deterministic string seed: {SEED}")


# ============================================================================
# SETUP - Generate synthetic log files
# ============================================================================


def generate_log_entry(timestamp, level, message):
  """Generate a single log entry in standard format."""
  return f"[{timestamp}] {level:8s} | {message}"


def generate_log_file(file_number, base_dir, has_critical=False):
  """
  Generate a single log file with random entries.

  Args:
    file_number: File number (1-indexed)
    base_dir: Directory to write the log file
    has_critical: Whether this file should contain CRITICAL incidents

  Returns:
    List of critical incidents in this file (for validation)
  """
  filename = f"log_{file_number:03d}.txt"
  filepath = os.path.join(base_dir, filename)

  # Generate timestamp range for this log file
  base_time = datetime.now() - timedelta(days=NUM_ITERATIONS - file_number)

  entries = []
  critical_incidents = []

  # Generate 15-25 log entries per file
  num_entries = random.randint(15, 25)

  for i in range(num_entries):
    timestamp = base_time + timedelta(minutes=i * 5)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Randomly select log level and message
    level_roll = random.random()

    if has_critical and i == num_entries // 2:
      # Inject a CRITICAL incident in the middle of the file
      level = "CRITICAL"
      messages = [
        "Unauthorized access attempt detected from IP 192.168.1.100",
        "SQL injection attempt blocked in user login form",
        "Privilege escalation attempt by user 'guest' denied",
        "Malicious file upload detected and quarantined",
        "Brute force attack detected on admin account",
        "Data exfiltration attempt blocked by firewall",
        "Ransomware signature detected in uploaded file",
        "Cross-site scripting (XSS) attack prevented",
        "Directory traversal attempt detected and blocked",
        "Suspicious process execution blocked by security policy",
      ]
      message = random.choice(messages)
      critical_incidents.append({"file": filename, "timestamp": timestamp_str, "message": message})
    elif level_roll < 0.6:
      # 60% INFO
      level = "INFO"
      messages = [
        "User login successful",
        "Database connection established",
        "Cache cleared successfully",
        "Scheduled backup completed",
        "API request processed in 45ms",
        "Session created for user",
        "File uploaded successfully",
        "Configuration reloaded",
        "Health check passed",
        "Metrics collected and sent",
      ]
      message = random.choice(messages)
    elif level_roll < 0.85:
      # 25% WARNING
      level = "WARNING"
      messages = [
        "High memory usage detected (85%)",
        "Slow database query (2.3s)",
        "Disk space running low (15% remaining)",
        "Failed login attempt (incorrect password)",
        "API rate limit approaching threshold",
        "SSL certificate expires in 30 days",
        "Deprecated API endpoint called",
        "Unusually high CPU usage detected",
        "Cache miss rate above threshold",
        "Connection pool nearly exhausted",
      ]
      message = random.choice(messages)
    else:
      # 15% ERROR
      level = "ERROR"
      messages = [
        "Failed to connect to external service",
        "Database transaction rolled back",
        "File not found: /tmp/missing.txt",
        "Network timeout after 30 seconds",
        "Invalid JSON payload received",
        "Permission denied when accessing file",
        "Out of memory exception caught",
        "Null pointer exception in module X",
        "Failed to send email notification",
        "Configuration file parsing error",
      ]
      message = random.choice(messages)

    entries.append(generate_log_entry(timestamp_str, level, message))

  # Write the log file
  with open(filepath, "w") as f:
    f.write("\n".join(entries) + "\n")

  return critical_incidents


def create_sample_logs(base_path, num_files):
  """
  Create a set of synthetic log files for testing.

  Args:
    base_path: Directory to create log files in
    num_files: Number of log files to generate

  Returns:
    Dictionary mapping file numbers to critical incidents
  """
  info(f"Generating {num_files} log files...")

  # Determine which files should have CRITICAL incidents
  # Distribute them somewhat evenly (roughly 20% of files)
  critical_file_numbers = set()
  num_critical_files = max(1, num_files // 5)  # At least 1, roughly 20%

  # Ensure even distribution
  step = num_files // num_critical_files
  for i in range(num_critical_files):
    file_num = (i * step) + random.randint(1, step // 2 if step > 2 else 1)
    file_num = min(file_num, num_files)  # Don't exceed total files
    critical_file_numbers.add(file_num)

  # Generate all log files
  expected_incidents = {}

  for file_num in range(1, num_files + 1):
    has_critical = file_num in critical_file_numbers
    incidents = generate_log_file(file_num, base_path, has_critical)

    if incidents:
      expected_incidents[file_num] = incidents

    # Progress indicator
    if file_num % 20 == 0:
      info(f"  Generated {file_num}/{num_files} files...")

  total_incidents = sum(len(incidents) for incidents in expected_incidents.values())
  info(f"✓ Created {num_files} log files")
  info(f"✓ Injected {total_incidents} CRITICAL incidents across {len(expected_incidents)} files")

  return expected_incidents


# ============================================================================
# INTERNAL EXECUTION ANALYSIS
# ============================================================================


def find_transcript_files(agent_id=None):
  """
  Find transcript files generated by AUTONOMY_TRANSCRIPTS_RAW_ONLY=1.

  Returns:
    List of transcript file paths (JSONL format)
  """
  transcript_paths = []

  # Common transcript locations
  search_dirs = [
    os.path.expanduser("~/.autonomy/transcripts"),
    "/tmp/autonomy_transcripts",
    os.getenv("AUTONOMY_TRANSCRIPTS_DIR"),
  ]

  # Also check current directory and temp
  search_dirs.extend(
    [
      os.path.join(tempfile.gettempdir(), "autonomy_transcripts"),
      os.path.join(os.getcwd(), ".autonomy", "transcripts"),
    ]
  )

  # Deduplicate search directories (e.g., if AUTONOMY_TRANSCRIPTS_DIR=/tmp/autonomy_transcripts)
  seen_dirs = set()
  unique_search_dirs = []
  for base_dir in search_dirs:
    if base_dir and base_dir not in seen_dirs:
      seen_dirs.add(base_dir)
      unique_search_dirs.append(base_dir)

  for base_dir in unique_search_dirs:
    if not os.path.exists(base_dir):
      continue

    for root, _, files in os.walk(base_dir):
      for f in files:
        if f.endswith(".jsonl"):
          # If agent_id specified, only include matching transcripts
          if agent_id is None or agent_id in f or agent_id in root:
            transcript_paths.append(os.path.join(root, f))

  return transcript_paths


def analyze_internal_execution(transcript_files, num_files, project_path):
  """
  Analyze internal multi-turn behavior from JSONL transcripts.

  Validates:
  - Sequential file access order
  - Completeness (all files accessed)
  - No fabricated file mentions
  - Tool call patterns

  Args:
    transcript_files: List of JSONL transcript file paths
    num_files: Expected number of files to process
    project_path: Base directory containing log files

  Returns:
    Dictionary with internal execution metrics
  """
  results = {
    "files_accessed": [],  # Ordered list of file numbers accessed
    "unique_files": set(),  # Set of unique file numbers
    "duplicates": [],  # Files accessed multiple times
    "gaps": [],  # Expected files never accessed
    "total_tool_calls": 0,  # Total filesystem tool calls
    "file_read_calls": 0,  # Specific read operations
    "suspect_skips": [],  # Large jumps in sequence (>10 files)
    "sequence_monotonic": True,  # Access order is non-decreasing
    "total_events": 0,  # Total JSONL events processed
    "fabricated_files": set(),  # Files mentioned but never read
  }

  if not transcript_files:
    info("⚠ No transcript files found - skipping internal execution analysis")
    return results

  info(f"Analyzing {len(transcript_files)} transcript file(s)...")

  # Parse all transcript files
  for transcript_path in transcript_files:
    try:
      with open(transcript_path, "r") as f:
        for line in f:
          if not line.strip():
            continue

          try:
            event = json.loads(line)
            results["total_events"] += 1

            # Only look at assistant messages with tool calls (not tool responses)
            # This avoids counting file references in tool response content
            if event.get("role") == "assistant" and "tool_calls" in event:
              tool_calls = event.get("tool_calls", [])

              for tool_call in tool_calls:
                # Get the function name and arguments
                function_info = tool_call.get("function", {})
                function_name = function_info.get("name", "")
                arguments_str = function_info.get("arguments", "")

                # Only look for log file references in read_file calls
                if function_name == "read_file":
                  matches = re.findall(r"log_(\d{3})\.txt", arguments_str)
                  for match in matches:
                    file_num = int(match)
                    results["file_read_calls"] += 1
                    results["files_accessed"].append(file_num)

                    if file_num in results["unique_files"]:
                      results["duplicates"].append(file_num)
                    results["unique_files"].add(file_num)

                # Count all tool calls for statistics
                results["total_tool_calls"] += 1

          except json.JSONDecodeError:
            continue

    except Exception as e:
      info(f"⚠ Error reading transcript {transcript_path}: {e}")
      continue

  # Analyze sequence
  expected = set(range(1, num_files + 1))
  results["gaps"] = sorted(list(expected - results["unique_files"]))

  # Check monotonicity (files accessed in order)
  prev = 0
  for file_num in results["files_accessed"]:
    if file_num < prev:
      results["sequence_monotonic"] = False
      break
    prev = file_num

  # Detect suspect skips (jumps > 10 files)
  ordered = sorted(results["unique_files"])
  prev = None
  for file_num in ordered:
    if prev is not None and file_num - prev > 10:
      results["suspect_skips"].append((prev, file_num))
    prev = file_num

  info(f"✓ Analyzed {results['total_events']} transcript events")
  info(f"✓ Found {results['file_read_calls']} file read operations")
  info(f"✓ Unique files accessed: {len(results['unique_files'])}/{num_files}")

  return results


def generate_internal_execution_report(internal_results, output_path):
  """Generate a report on internal execution analysis."""
  report = f"""# Internal Execution Analysis

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Transcript Events
- Total events processed: {internal_results["total_events"]}
- Total tool calls: {internal_results["total_tool_calls"]}
- File read operations: {internal_results["file_read_calls"]}

## File Access Pattern
- Unique files accessed: {len(internal_results["unique_files"])}
- Sequence monotonic: {"✓ Yes" if internal_results["sequence_monotonic"] else "❌ No"}
- Duplicate accesses: {len(set(internal_results["duplicates"]))} files
- Missing files (gaps): {len(internal_results["gaps"])}

### Details
"""

  if internal_results["gaps"]:
    gap_preview = internal_results["gaps"][:20]
    report += f"\n**Missing files:** {gap_preview}{'...' if len(internal_results['gaps']) > 20 else ''}\n"
  else:
    report += "\n**Missing files:** None\n"

  if internal_results["duplicates"]:
    dup_preview = sorted(set(internal_results["duplicates"]))[:10]
    report += f"**Duplicate accesses:** {dup_preview}{'...' if len(set(internal_results['duplicates'])) > 10 else ''}\n"

  if internal_results["suspect_skips"]:
    report += f"\n**Suspect skips (jumps > 10):** {internal_results['suspect_skips']}\n"

  if internal_results["fabricated_files"]:
    fab_preview = sorted(list(internal_results["fabricated_files"]))[:10]
    report += f"\n**Fabricated file mentions:** {fab_preview}{'...' if len(internal_results['fabricated_files']) > 10 else ''}\n"

  report += f"""
## Verdict

"""

  if (
    internal_results["sequence_monotonic"] and not internal_results["gaps"] and not internal_results["fabricated_files"]
  ):
    report += "✓ **PASS** - Agent executed sequentially, accessed all files, no fabrications detected.\n"
  else:
    report += "❌ **FAIL** - Issues detected in internal execution pattern.\n"

  report += "\n---\n*Generated by 012.py internal execution analyzer*\n"

  with open(output_path, "w") as f:
    f.write(report)


# ============================================================================
# VALIDATION
# ============================================================================


def validate_results(project_path, expected_incidents, num_files):
  """
  Validate the agent's output against expected results using JSON summary.

  Args:
    project_path: Path to the project directory
    expected_incidents: Dictionary of expected critical incidents
    num_files: Total number of files that should have been processed

  Returns:
    Validation results dictionary
  """
  info("-" * 80)
  info("VALIDATION: Checking Agent Output")
  info("-" * 80)
  info("")

  expected_incident_count = sum(len(incidents) for incidents in expected_incidents.values())

  results = {
    "total_files": num_files,
    "expected_incidents": expected_incident_count,
    "found_incidents": 0,
    "missing_incidents": [],
    "fabricated_incidents": [],
    "files_processed": set(),
    "checkpoints_found": [],
    "format_consistent": True,
    "json_valid": False,
    "success": False,
  }

  # Check if JSON summary exists (required)
  json_path = os.path.join(project_path, "incident_summary.json")
  if not os.path.exists(json_path):
    info("❌ FAIL: incident_summary.json not found")
    results["success"] = False
    return results

  info("✓ Found incident_summary.json")

  # Check if markdown report exists (optional but expected)
  report_path = os.path.join(project_path, "incident_report.md")
  if os.path.exists(report_path):
    info("✓ Found incident_report.md")
  else:
    info("⚠ incident_report.md not found (optional)")

  # Load and parse JSON summary
  try:
    with open(json_path, "r") as f:
      summary = json.load(f)
    results["json_valid"] = True
    info("✓ JSON is valid")
  except json.JSONDecodeError as e:
    info(f"❌ FAIL: Invalid JSON - {e}")
    results["success"] = False
    return results

  # Extract data from JSON - be flexible with key names
  # Try various possible key names for files processed
  files_processed = summary.get("files_processed", []) or summary.get("files", []) or []

  # Check nested in 'summary' object
  if not files_processed and "summary" in summary:
    nested_summary = summary["summary"]
    files_processed = nested_summary.get("files_processed", []) or nested_summary.get("files", []) or []

  # If no explicit files list, check verification/validation section
  if not files_processed:
    # Try both "verification" and "validation" keys
    validation_section = summary.get("verification") or summary.get("validation") or {}

    # Check for actual_files or actual_files_processed count
    actual_count = (
      validation_section.get("actual_files")
      or validation_section.get("actual_files_processed")
      or validation_section.get("expected_files")
      or 0
    )

    # If all_files_processed is true, generate the list
    all_processed = (
      validation_section.get("all_files_processed") or validation_section.get("processing_complete") or False
    )

    if all_processed and actual_count:
      files_processed = [f"log_{i:03d}.txt" for i in range(1, actual_count + 1)]

    # Handle keys like "all_20_files_processed"
    if not files_processed:
      for key in validation_section:
        if key.startswith("all_") and key.endswith("_files_processed") and validation_section[key]:
          files_processed = [f"log_{i:03d}.txt" for i in range(1, num_files + 1)]
          break

  # Also check summary.total_files_processed with validation.all_files_processed
  if not files_processed and "summary" in summary:
    nested_summary = summary["summary"]
    total_files = nested_summary.get("total_files_processed", 0)
    validation_section = summary.get("verification") or summary.get("validation") or {}
    all_processed = (
      validation_section.get("all_files_processed") or validation_section.get("processing_complete") or False
    )
    if total_files and all_processed:
      files_processed = [f"log_{i:03d}.txt" for i in range(1, total_files + 1)]

  # Try various possible key names for incidents
  incidents = (
    summary.get("incidents", []) or summary.get("critical_incidents", []) or summary.get("security_incidents", []) or []
  )

  # Try various possible key names for checkpoints
  checkpoints = (
    summary.get("checkpoints", [])
    or summary.get("processing_checkpoints", [])
    or summary.get("progress_checkpoints", [])
    or []
  )

  # Check nested in 'statistics' object
  if not checkpoints and "statistics" in summary:
    stats = summary["statistics"]
    stats_checkpoints = stats.get("checkpoints", {})
    if isinstance(stats_checkpoints, dict):
      # Convert dict format to list format
      checkpoints = list(stats_checkpoints.values())
    elif isinstance(stats_checkpoints, list):
      checkpoints = stats_checkpoints

  # Convert file names to numbers for comparison
  for filename in files_processed:
    match = re.match(r"log_(\d{3})\.txt", filename)
    if match:
      results["files_processed"].add(int(match.group(1)))

  results["found_incidents"] = len(incidents)

  # Extract checkpoint file counts - handle various formats
  checkpoint_counts = []
  for cp in checkpoints:
    # Try various key names for the count
    count = (
      cp.get("files_completed")
      or cp.get("files_processed")
      or cp.get("checkpoint", 0) * CHECKPOINT_INTERVAL  # If using checkpoint number
    )
    if isinstance(count, str) and "to" in count:
      # Handle format like "log_001.txt to log_010.txt"
      match = re.search(r"log_(\d{3})\.txt$", count)
      if match:
        count = int(match.group(1))
    if isinstance(count, int) and count > 0:
      checkpoint_counts.append(count)
  results["checkpoints_found"] = checkpoint_counts

  info(f"✓ Files processed: {len(results['files_processed'])}/{num_files}")
  info(f"✓ Incidents found: {results['found_incidents']}/{results['expected_incidents']}")
  info(f"✓ Checkpoints found: {len(results['checkpoints_found'])}")
  info("")

  # Detailed validation
  info("Detailed Validation:")

  # Check completeness
  missing_files = set(range(1, num_files + 1)) - results["files_processed"]
  if missing_files:
    info(f"  ⚠ Missing files: {sorted(missing_files)[:10]}{'...' if len(missing_files) > 10 else ''}")
  else:
    info(f"  ✓ All {num_files} files were processed")

  # Check incident accuracy
  incident_delta = results["found_incidents"] - results["expected_incidents"]
  if incident_delta == 0:
    info(f"  ✓ Exact incident count match ({results['expected_incidents']})")
  elif incident_delta > 0:
    info(f"  ⚠ Found {incident_delta} extra incidents (possible false positives)")
  else:
    info(f"  ⚠ Missing {-incident_delta} incidents (false negatives)")

  # Check checkpoints
  expected_checkpoints = list(range(CHECKPOINT_INTERVAL, num_files + 1, CHECKPOINT_INTERVAL))
  if num_files not in expected_checkpoints:
    expected_checkpoints.append(num_files)

  if results["checkpoints_found"]:
    info(f"  ✓ Found {len(results['checkpoints_found'])} checkpoints")
  else:
    info(f"  ⚠ No checkpoints found (expected around {len(expected_checkpoints)})")

  # Validate incident details against expected
  expected_files_with_incidents = set(expected_incidents.keys())
  found_files_with_incidents = set()
  for incident in incidents:
    # Handle various key names for the file field
    filename = incident.get("file") or incident.get("source_file") or incident.get("filename") or ""
    match = re.match(r"log_(\d{3})\.txt", filename)
    if match:
      found_files_with_incidents.add(int(match.group(1)))

  missing_incident_files = expected_files_with_incidents - found_files_with_incidents
  fabricated_incident_files = found_files_with_incidents - expected_files_with_incidents

  if missing_incident_files:
    results["missing_incidents"] = sorted(missing_incident_files)
    info(f"  ⚠ Missing incidents from files: {results['missing_incidents']}")

  if fabricated_incident_files:
    results["fabricated_incidents"] = sorted(fabricated_incident_files)
    info(f"  ⚠ Fabricated incidents in files: {results['fabricated_incidents']}")

  if not missing_incident_files and not fabricated_incident_files:
    info(f"  ✓ All incidents correctly identified")

  info("")

  # Determine overall success
  results["success"] = (
    results["json_valid"]
    and len(results["files_processed"]) >= num_files * 0.95  # At least 95% completion
    and abs(incident_delta) <= max(1, results["expected_incidents"] * 0.1)  # Within 10% accuracy or 1
  )

  return results


def generate_validation_report(results, internal_results, output_path):
  """Generate a validation report summarizing the test results."""
  report = f"""# Persistence Test Validation Report

**Test Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Iterations:** {results["total_files"]}
**Result:** {"✓ PASS" if results["success"] else "❌ FAIL"}

## Summary

- **Files Processed:** {len(results["files_processed"])}/{results["total_files"]} ({len(results["files_processed"]) / results["total_files"] * 100:.1f}%)
- **Expected Incidents:** {results["expected_incidents"]}
- **Found Incidents:** {results["found_incidents"]}
- **Accuracy:** {results["found_incidents"] / max(results["expected_incidents"], 1) * 100:.1f}%
- **Checkpoints Found:** {len(results["checkpoints_found"])}

## Internal Execution (from Transcripts)

- **Files Actually Read:** {len(internal_results.get("unique_files", set()))} (from transcript analysis)
- **Sequential Access:** {"✓ Yes" if internal_results.get("sequence_monotonic", False) else "❌ No"}
- **Missing Files:** {len(internal_results.get("gaps", []))}
- **Fabricated Mentions:** {len(internal_results.get("fabricated_files", set()))}

## Detailed Results

### Completion
{"✓ All files processed successfully" if len(results["files_processed"]) == results["total_files"] else f"⚠ Missing {results['total_files'] - len(results['files_processed'])} files"}

### Accuracy
{"✓ Perfect incident detection" if results["found_incidents"] == results["expected_incidents"] else f"⚠ Incident count mismatch (delta: {results['found_incidents'] - results['expected_incidents']})"}

### Checkpoints
{"✓ Progress checkpoints present" if results["checkpoints_found"] else "⚠ No checkpoints found"}
{f"Found checkpoints at: {results['checkpoints_found']}" if results["checkpoints_found"] else ""}

## Conclusion

{"The agent successfully maintained consistency and accuracy across all iterations." if results["success"] else "The agent showed signs of drift, missed files, or inaccurate reporting."}

---
*Generated by 012.py*
"""

  with open(output_path, "w") as f:
    f.write(report)

  info(f"✓ Validation report saved to: {output_path}")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================


async def main(node):
  info("=" * 80)
  info(f"EXAMPLE 012: Agent Persistence Test ({NUM_ITERATIONS} Iterations)")
  info("=" * 80)
  info("")

  # Create a temporary directory for our test
  temp_dir = tempfile.mkdtemp(prefix="persistence_test_")
  project_path = os.path.join(temp_dir, "logs")
  os.makedirs(project_path, exist_ok=True)

  try:
    info("-" * 80)
    info("SETUP: Generating Test Data")
    info("-" * 80)
    info("")

    # Generate synthetic log files
    expected_incidents = create_sample_logs(project_path, NUM_ITERATIONS)
    info("")

    info("-" * 80)
    info("SETUP: Initializing Agent")
    info("-" * 80)
    info("")

    # Create filesystem tools
    fs_tools = FilesystemTools(
      visibility="all",
      base_dir=project_path,
    )

    # Create the agent with detailed instructions
    agent = await Agent.start(
      node=node,
      name="log-analyzer",
      instructions=f"""
        You are a security log analyzer. Your task is to process a series of log
        files and extract CRITICAL security incidents.

        INSTRUCTIONS:
        1. Process log files in order: log_001.txt, log_002.txt, ..., log_{NUM_ITERATIONS:03d}.txt
        2. For each file, read it and identify any lines with "CRITICAL" level
        3. Extract: filename, timestamp, and incident description
        4. Create TWO output files:

        FILE 1: incident_report.md (human-readable report)
        - Format this however you like, but include all incidents found
        - Add checkpoint summaries every {CHECKPOINT_INTERVAL} files
        - Include a final summary

        FILE 2: incident_summary.json (machine-readable, REQUIRED)
        - This file MUST be valid JSON with this EXACT structure:
        {{
          "files_processed": ["log_001.txt", "log_002.txt", ...],
          "total_files": {NUM_ITERATIONS},
          "incidents": [
            {{"file": "log_XXX.txt", "timestamp": "YYYY-MM-DD HH:MM:SS", "description": "..."}}
          ],
          "checkpoints": [
            {{"files_completed": {CHECKPOINT_INTERVAL}, "incidents_so_far": N}},
            {{"files_completed": {CHECKPOINT_INTERVAL * 2}, "incidents_so_far": N}}
          ]
        }}

        CRITICAL REQUIREMENTS:
        - Process EVERY file from log_001.txt to log_{NUM_ITERATIONS:03d}.txt
        - Do NOT skip files even if they have no incidents
        - Do NOT fabricate incidents that aren't in the logs
        - The incident_summary.json file MUST be valid JSON
        - Add a checkpoint entry every {CHECKPOINT_INTERVAL} files
        - Be thorough and accurate

        After processing all files, double-check to ensure:
        - All {NUM_ITERATIONS} files are listed in files_processed
        - Only real CRITICAL incidents are in the incidents array
        - The JSON is valid and complete
      """,
      model=Model(name="claude-sonnet-4-v1"),
      tools=[fs_tools],
    )

    agent_id = await agent.identifier()
    info(f"✓ Agent created with ID: {agent_id}")
    info(f"✓ Agent workspace: {project_path}")
    info(f"✓ Files to process: {NUM_ITERATIONS}")
    info(f"✓ Checkpoint interval: every {CHECKPOINT_INTERVAL} files")
    info("")

    info("-" * 80)
    info("EXECUTION: Starting Persistence Test")
    info("-" * 80)
    info("")

    start_time = datetime.now()

    info(f"👤 User: Process all {NUM_ITERATIONS} log files and create an incident report.")
    info("")

    # Send the task to the agent
    response = await agent.send(
      f"Process all log files from log_001.txt through log_{NUM_ITERATIONS:03d}.txt. "
      f"Extract all CRITICAL incidents and create both incident_report.md and incident_summary.json "
      f"as specified in your instructions. Add checkpoint summaries every {CHECKPOINT_INTERVAL} files. "
      f"When complete, verify you processed all {NUM_ITERATIONS} files. "
      f"IMPORTANT: The incident_summary.json file must be valid JSON.",
      timeout=TIMEOUT,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    info(f"🤖 Agent: {response}")
    info("")
    info(f"⏱️  Execution time: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    info("")

    info("-" * 80)
    info("")

    # Validate the results
    validation_results = validate_results(project_path, expected_incidents, NUM_ITERATIONS)

    # Analyze internal execution from transcripts
    info("-" * 80)
    info("INTERNAL EXECUTION: Analyzing Transcripts")
    info("-" * 80)
    info("")

    transcript_files = find_transcript_files(agent_id=agent_id)
    internal_results = analyze_internal_execution(transcript_files, NUM_ITERATIONS, project_path)

    # Detect fabricated file mentions (in report but never read)
    report_path = os.path.join(project_path, "incident_report.md")
    if os.path.exists(report_path):
      with open(report_path, "r") as f:
        report_content = f.read()
      mentioned_files = set(int(num) for num in re.findall(r"log_(\d{3})\.txt", report_content))
    else:
      mentioned_files = set()

    accessed_files = internal_results.get("unique_files", set())
    fabricated = mentioned_files - accessed_files
    internal_results["fabricated_files"] = fabricated

    if fabricated:
      info(f"⚠ Files mentioned but never accessed: {sorted(list(fabricated))[:10]}")
    else:
      info("✓ No fabricated file mentions detected")

    info("")

    # Generate internal execution report
    internal_report_path = os.path.join(project_path, "internal_execution_analysis.md")
    generate_internal_execution_report(internal_results, internal_report_path)
    info(f"✓ Internal execution report saved to: {internal_report_path}")
    info("")

    # Generate validation report
    validation_report_path = os.path.join(project_path, "validation_results.md")
    generate_validation_report(validation_results, internal_results, validation_report_path)
    info("")

    # Save metrics as JSON
    # Enhanced success criteria including internal execution
    overall_success = (
      validation_results["success"]
      and internal_results.get("sequence_monotonic", False)
      and len(internal_results.get("gaps", [])) == 0
      and len(internal_results.get("fabricated_files", set())) == 0
    )

    metrics = {
      "num_iterations": NUM_ITERATIONS,
      "execution_time_seconds": duration,
      "files_processed": len(validation_results["files_processed"]),
      "files_actually_read": len(internal_results.get("unique_files", set())),
      "expected_incidents": validation_results["expected_incidents"],
      "found_incidents": validation_results["found_incidents"],
      "checkpoints_found": len(validation_results["checkpoints_found"]),
      "sequence_monotonic": internal_results.get("sequence_monotonic", False),
      "missing_files": len(internal_results.get("gaps", [])),
      "fabricated_files": len(internal_results.get("fabricated_files", set())),
      "output_validation_success": validation_results["success"],
      "internal_execution_success": (
        internal_results.get("sequence_monotonic", False)
        and len(internal_results.get("gaps", [])) == 0
        and len(internal_results.get("fabricated_files", set())) == 0
      ),
      "overall_success": overall_success,
      "timestamp": datetime.now().isoformat(),
    }

    metrics_path = os.path.join(project_path, "metrics.json")
    with open(metrics_path, "w") as f:
      json.dump(metrics, f, indent=2)

    info(f"✓ Metrics saved to: {metrics_path}")
    info("")

    # Show sample of incident report
    info("-" * 80)
    info("SAMPLE OUTPUT: Incident Report (first 800 characters)")
    info("-" * 80)
    info("")

    if os.path.exists(report_path):
      with open(report_path, "r") as f:
        content = f.read()
        preview = content[:800] + ("..." if len(content) > 800 else "")
        for line in preview.split("\n"):
          info(f"  {line}")
    else:
      info("  (Incident report not found)")

    info("")
    info("-" * 80)
    info("")

    # Final summary
    info("=" * 80)
    info("TEST RESULTS")
    info("=" * 80)
    info("")

    if overall_success:
      info("✓✓✓ PERSISTENCE TEST PASSED ✓✓✓")
      info("")
      info("The agent successfully:")
      info(f"  • Processed {len(validation_results['files_processed'])}/{NUM_ITERATIONS} log files (output)")
      info(f"  • Actually read {len(internal_results.get('unique_files', set()))}/{NUM_ITERATIONS} files (transcript)")
      info(
        f"  • Identified {validation_results['found_incidents']}/{validation_results['expected_incidents']} critical incidents"
      )
      info(f"  • Maintained sequential access order")
      info(f"  • No fabricated file mentions")
      info(f"  • Maintained consistent format throughout")
      info(f"  • Completed in {duration / 60:.1f} minutes")
    else:
      info("❌❌❌ PERSISTENCE TEST FAILED ❌❌❌")
      info("")
      info("Issues detected:")

      # Output validation issues
      missing_files = NUM_ITERATIONS - len(validation_results["files_processed"])
      if missing_files > 0:
        info(f"  • Missed {missing_files} files in output")

      incident_delta = validation_results["found_incidents"] - validation_results["expected_incidents"]
      if incident_delta != 0:
        info(
          f"  • Incident accuracy off by {abs(incident_delta)} ({'+' if incident_delta > 0 else ''}{incident_delta})"
        )

      if not validation_results["checkpoints_found"]:
        info(f"  • No checkpoint summaries found")

      # Internal execution issues
      if not internal_results.get("sequence_monotonic", False):
        info(f"  • Non-sequential file access detected")

      if internal_results.get("gaps"):
        info(f"  • {len(internal_results['gaps'])} files never accessed (transcript)")

      if internal_results.get("fabricated_files"):
        info(f"  • {len(internal_results['fabricated_files'])} fabricated file mentions")

    info("")
    info("Key Metrics:")
    info(f"  • Total iterations: {NUM_ITERATIONS}")
    info(f"  • Output completion rate: {len(validation_results['files_processed']) / NUM_ITERATIONS * 100:.1f}%")
    info(f"  • Internal access rate: {len(internal_results.get('unique_files', set())) / NUM_ITERATIONS * 100:.1f}%")
    info(
      f"  • Accuracy rate: {validation_results['found_incidents'] / max(validation_results['expected_incidents'], 1) * 100:.1f}%"
    )
    info(f"  • Sequential access: {'Yes' if internal_results.get('sequence_monotonic', False) else 'No'}")
    info(f"  • Fabricated mentions: {len(internal_results.get('fabricated_files', set()))}")
    info(f"  • Execution time: {duration:.1f}s ({duration / NUM_ITERATIONS:.2f}s per iteration)")
    info(f"  • Checkpoints: {len(validation_results['checkpoints_found'])}")
    info("")

    info("Generated Files:")
    info(f"  • incident_report.md - Agent's incident tracking report")
    info(f"  • validation_results.md - Test validation summary")
    info(f"  • internal_execution_analysis.md - Internal execution analysis")
    info(f"  • metrics.json - Performance metrics")
    info(f"  • log_001.txt through log_{NUM_ITERATIONS:03d}.txt - Test data")
    info("")

    info("To inspect results:")
    info(f"  cat {report_path}")
    info(f"  cat {validation_report_path}")
    info(f"  cat {internal_report_path}")
    info(f"  cat {metrics_path}")
    info("")

    info(f"Test artifacts preserved at: {temp_dir}")
    info("")

  finally:
    # Preserve artifacts for inspection
    info("✓ Test artifacts preserved for inspection")
    info("")


Node.start(main)
