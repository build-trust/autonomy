"""
Example 010: Code Review Assistant with Filesystem Tools

This example demonstrates a complex multi-step task using an agent with
filesystem capabilities. The agent acts as a code review assistant that:

1. Analyzes code files in a project directory
2. Identifies potential issues and improvements
3. Generates a comprehensive review report
4. Tracks issues across multiple review sessions
5. Creates summary documents with recommendations

This showcases:
- Filesystem tools with 'all' visibility (direct project access)
- Multi-step reasoning and task decomposition
- Reading, writing, and editing files
- Searching through file contents
- Organizing information in a structured way

Note: This example uses visibility="all" so the agent can access files directly
in the temporary project directory. For production use cases with persistent
storage, consider using visibility="scope" or "conversation" for better isolation.

RECOMMENDED: Run with transcript mode for detailed output:
  AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
  AUTONOMY_TRANSCRIPTS_DIR=/tmp/transcripts \
  uv run --active examples/010.py


"""

from autonomy import Agent, Model, Node, info
from autonomy.tools.filesystem import FilesystemTools
import os
import tempfile
import shutil


# ============================================================================
# SETUP - Create a sample project to review
# ============================================================================


def create_sample_project(base_path: str):
  """Create a sample Python project with various code quality issues."""

  # Create project structure
  os.makedirs(os.path.join(base_path, "src"), exist_ok=True)
  os.makedirs(os.path.join(base_path, "tests"), exist_ok=True)

  # File 1: main.py with various issues
  main_py = """
import sys
import os
import time

# TODO: This function needs better error handling
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item  # Could use +=
    return total

# FIXME: Security issue - no input validation
def process_user_input(user_data):
    exec(user_data)  # Dangerous!
    return "Processed"

class DataProcessor:
    def __init__(self):
        self.data = []
        self.count = 0

    # Missing docstring
    def add_item(self, item):
        self.data.append(item)
        self.count = self.count + 1

    def get_stats(self):
        # Inefficient: recalculating on every call
        avg = sum(self.data) / len(self.data) if self.data else 0
        return {
            'count': self.count,
            'average': avg,
            'total': sum(self.data)
        }

# No error handling
def read_config_file(filename):
    with open(filename) as f:
        return f.read()

if __name__ == '__main__':
    # Hardcoded path - not portable
    config = read_config_file('/tmp/config.txt')
    processor = DataProcessor()
    processor.add_item(10)
    processor.add_item(20)
    print(processor.get_stats())
"""

  # File 2: utils.py with additional issues
  utils_py = """
import requests

# Missing type hints
def fetch_data(url):
    response = requests.get(url)  # No timeout specified
    return response.json()  # No error checking

# Global variable - could cause issues
CACHE = {}

def cache_result(key, value):
    CACHE[key] = value  # No size limit - memory leak risk

def get_cached(key):
    return CACHE.get(key)

# Deprecated pattern
class OldStyleClass:
    pass

def validate_email(email):
    # Weak validation - use a library instead
    return '@' in email and '.' in email

# SQL injection vulnerability
def query_database(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    # execute(query)  # Unsafe!
    pass
"""

  # File 3: A test file with minimal coverage
  test_main_py = """
import unittest
from src.main import calculate_total

class TestCalculateTotal(unittest.TestCase):
    def test_basic(self):
        result = calculate_total([1, 2, 3])
        self.assertEqual(result, 6)

    # Missing tests for edge cases:
    # - Empty list
    # - Negative numbers
    # - Large numbers
    # - Non-numeric inputs

if __name__ == '__main__':
    unittest.main()
"""

  # File 4: README
  readme_md = """
# Sample Project

A simple Python project for demonstration purposes.

## Installation

```
pip install -r requirements.txt
```

## Usage

Run the main script:
```
python src/main.py
```

## TODO
- Add proper documentation
- Improve error handling
- Add more tests
- Security audit needed
"""

  # Write all files
  with open(os.path.join(base_path, "src", "main.py"), "w") as f:
    f.write(main_py)

  with open(os.path.join(base_path, "src", "utils.py"), "w") as f:
    f.write(utils_py)

  with open(os.path.join(base_path, "tests", "test_main.py"), "w") as f:
    f.write(test_main_py)

  with open(os.path.join(base_path, "README.md"), "w") as f:
    f.write(readme_md)

  info(f"✓ Created sample project at: {base_path}")
  info(f"  - src/main.py (main application)")
  info(f"  - src/utils.py (utility functions)")
  info(f"  - tests/test_main.py (test suite)")
  info(f"  - README.md (documentation)")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================


async def main(node):
  info("=" * 80)
  info("EXAMPLE 010: Code Review Assistant with Filesystem Tools")
  info("=" * 80)
  info("")

  # Create a temporary directory for our sample project
  temp_dir = tempfile.mkdtemp(prefix="code_review_")
  project_path = os.path.join(temp_dir, "sample_project")
  os.makedirs(project_path, exist_ok=True)

  try:
    info("-" * 80)
    info("SETUP: Creating Sample Project")
    info("-" * 80)
    info("")

    create_sample_project(project_path)
    info("")

    info("-" * 80)
    info("SETUP: Initializing Code Review Agent")
    info("-" * 80)
    info("")

    # Create filesystem tools with all visibility
    # This gives the agent access to all files in the project directory
    fs_tools = FilesystemTools(
      visibility="all",
      base_dir=project_path,
    )

    # Create the code review agent
    agent = await Agent.start(
      node=node,
      name="code-reviewer",
      instructions="""
        You are an expert code review assistant specializing in Python.
        Your goal is to help developers improve their code quality by:

        1. Analyzing code files for potential issues
        2. Identifying security vulnerabilities
        3. Suggesting performance improvements
        4. Checking for code style and best practices
        5. Evaluating test coverage
        6. Documenting findings in structured reports

        When conducting a code review:
        - Start by exploring the project structure
        - Read and analyze each code file systematically
        - Look for common issues: security, performance, maintainability
        - Check for TODO/FIXME comments that indicate known issues
        - Evaluate test coverage and quality
        - Create a comprehensive review report with clear sections
        - Prioritize findings by severity (Critical, High, Medium, Low)
        - Provide specific, actionable recommendations

        You have access to filesystem tools to read files, search for patterns,
        and create reports. Use these tools effectively to conduct thorough reviews.

        Always be constructive and educational in your feedback. Explain WHY
        something is an issue and HOW to fix it.
      """,
      model=Model(name="claude-sonnet-4-v1"),
      tools=[fs_tools],
    )

    agent_id = await agent.identifier()
    info(f"✓ Agent created with ID: {agent_id}")
    info(f"✓ Agent has access to: {project_path}")
    info(f"✓ Filesystem visibility: all (full project access)")
    info("")

    info("-" * 80)
    info("REVIEW SESSION 1: Initial Code Review")
    info("-" * 80)
    info("")

    # Set timeout for all agent tasks
    # Code review involves multiple steps: exploring files, reading content,
    # analyzing code, searching for patterns, and generating reports.
    # 20 minutes allows the agent to complete thorough analysis.
    TIMEOUT = 1200  # 20 minutes (1200 seconds)

    # Task 1: Initial exploration and overview
    info("👤 User: Please review the code in this project and create a comprehensive")
    info("         review report. Start by exploring the project structure.")
    info("")

    response1 = await agent.send(
      "Please review the code in this project. Start by exploring the project "
      "structure, then analyze each file for issues. Create a comprehensive "
      "review report that includes: 1) Project overview, 2) Critical issues found, "
      "3) Security concerns, 4) Performance suggestions, 5) Code quality recommendations, "
      "and 6) Test coverage assessment. Save the report as 'review_report.md'.",
      timeout=TIMEOUT
    )

    info(f"🤖 Agent: {response1}")
    info("")

    info("-" * 80)
    info("")

    # Task 2: Focus on specific security issues
    info("👤 User: Can you create a separate document listing all security")
    info("         vulnerabilities you found, ranked by severity?")
    info("")

    response2 = await agent.send(
      "Create a separate document called 'security_issues.md' that lists all "
      "security vulnerabilities you found, ranked by severity (Critical, High, "
      "Medium, Low). For each issue, include: the file and line number, "
      "description of the vulnerability, potential impact, and recommended fix.",
      timeout=TIMEOUT
    )

    info(f"🤖 Agent: {response2}")
    info("")

    info("-" * 80)
    info("")

    # Task 3: Track action items
    info("👤 User: Generate an action items checklist based on your findings.")
    info("")

    response3 = await agent.send(
      "Based on all the issues you've identified, create an 'action_items.md' "
      "file with a prioritized checklist of tasks to improve this codebase. "
      "Group items by category (Security, Performance, Testing, Code Quality) "
      "and use markdown checkboxes. Put the most critical items at the top.",
      timeout=TIMEOUT
    )

    info(f"🤖 Agent: {response3}")
    info("")

    info("-" * 80)
    info("REVIEW SESSION 2: Follow-up Analysis")
    info("-" * 80)
    info("")

    # Task 4: Code metrics
    info("👤 User: Can you analyze the codebase and give me some metrics?")
    info("")

    response4 = await agent.send(
      "Analyze the codebase and create a 'metrics.md' file with statistics like: "
      "total lines of code, number of functions, number of classes, TODO/FIXME "
      "count, test coverage estimate, and number of files. Also identify which "
      "files are the most complex and might need refactoring.",
      timeout=TIMEOUT
    )

    info(f"🤖 Agent: {response4}")
    info("")

    info("-" * 80)
    info("")

    # Task 5: Generate summary for stakeholders
    info("👤 User: Create an executive summary for non-technical stakeholders.")
    info("")

    response5 = await agent.send(
      "Create an 'executive_summary.md' file that presents your findings in "
      "non-technical language suitable for project managers and stakeholders. "
      "Include: overall code quality rating (1-10), top 3 concerns, estimated "
      "effort to address critical issues, and recommended next steps. Keep it "
      "concise and business-focused.",
      timeout=TIMEOUT
    )

    info(f"🤖 Agent: {response5}")
    info("")

    info("-" * 80)
    info("REVIEW COMPLETE: Generated Documentation")
    info("-" * 80)
    info("")

    # List all generated files
    info("📁 Review artifacts created:")
    info("")
    for root, dirs, files in os.walk(project_path):
      # Skip source directories, only show generated reports
      rel_root = os.path.relpath(root, project_path)
      if rel_root == "." or rel_root.startswith("src") or rel_root.startswith("tests"):
        for file in sorted(files):
          if file.endswith(".md") and file != "README.md":
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            info(f"  ✓ {os.path.join(rel_root, file) if rel_root != '.' else file} ({size} bytes)")

    info("")
    info("-" * 80)
    info("")

    # Show a sample of one of the generated reports
    info("📄 Sample Output: Executive Summary")
    info("")
    exec_summary_path = os.path.join(project_path, "executive_summary.md")
    if os.path.exists(exec_summary_path):
      with open(exec_summary_path, "r") as f:
        content = f.read()
        # Show first 500 characters
        preview = content[:500] + ("..." if len(content) > 500 else "")
        for line in preview.split("\n"):
          info(f"  {line}")
    else:
      info("  (Executive summary not found)")

    info("")
    info("-" * 80)
    info("")

    # Final summary
    info("=" * 80)
    info("EXAMPLE SUMMARY")
    info("=" * 80)
    info("")
    info("This example demonstrated:")
    info("  ✓ Complex multi-step task execution")
    info("  ✓ Filesystem tools for reading and analyzing code")
    info("  ✓ Creating structured documentation")
    info("  ✓ Multi-turn conversations with state preservation")
    info("  ✓ Agent reasoning across multiple code files")
    info("  ✓ Organizing information in different formats")
    info("")
    info("The agent successfully:")
    info("  • Explored the project structure")
    info("  • Identified security vulnerabilities")
    info("  • Analyzed code quality issues")
    info("  • Generated multiple review documents")
    info("  • Created actionable recommendations")
    info("  • Produced stakeholder-friendly summaries")
    info("")
    info("Key features used:")
    info("  • FilesystemTools with scope-level visibility")
    info("  • list_directory - exploring project structure")
    info("  • read_file - analyzing source code")
    info("  • write_file - creating review reports")
    info("  • search_in_files - finding patterns (TODO, FIXME, etc.)")
    info("")
    info(f"Review artifacts are available at: {project_path}")
    info("")
    info("To inspect the generated files:")
    info(f"  ls -lh {project_path}/*.md")
    info(f"  cat {project_path}/review_report.md")
    info(f"  cat {project_path}/security_issues.md")
    info("")

  finally:
    # Cleanup - commented out to preserve review artifacts
    info("Preserving temporary files for inspection...")
    info(f"✓ Review artifacts saved at: {temp_dir}")
    # try:
    #   shutil.rmtree(temp_dir)
    #   info(f"✓ Removed: {temp_dir}")
    # except Exception as e:
    #   info(f"⚠ Could not remove temp directory: {e}")
    info("")


Node.start(main)
