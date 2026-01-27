from autonomy import Agent, HttpServer, Model, Node, NodeDep, Zone
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional
import urllib.request
import os
import shutil
import zipfile
import re


# Language constants
LANGUAGE_PYTHON = "python"
LANGUAGE_NODE = "node"

PYTHON_EXTENSIONS = [".py"]
NODE_EXTENSIONS = [".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"]

NODE_BUILTINS = {
    'fs', 'path', 'http', 'https', 'crypto', 'os', 'util', 'stream',
    'events', 'buffer', 'url', 'querystring', 'child_process', 'cluster',
    'dgram', 'dns', 'net', 'readline', 'repl', 'tls', 'tty', 'vm', 'zlib',
    'assert', 'async_hooks', 'console', 'constants', 'domain', 'inspector',
    'module', 'perf_hooks', 'process', 'punycode', 'string_decoder',
    'timers', 'trace_events', 'v8', 'worker_threads',
    'node:fs', 'node:path', 'node:http', 'node:https', 'node:crypto', 'node:os',
    'node:util', 'node:stream', 'node:events', 'node:buffer', 'node:url',
    'node:querystring', 'node:child_process', 'node:cluster', 'node:dgram',
    'node:dns', 'node:net', 'node:readline', 'node:repl', 'node:tls', 'node:tty',
    'node:vm', 'node:zlib', 'node:assert', 'node:async_hooks', 'node:console',
    'node:constants', 'node:domain', 'node:inspector', 'node:module',
    'node:perf_hooks', 'node:process', 'node:punycode', 'node:string_decoder',
    'node:timers', 'node:trace_events', 'node:v8', 'node:worker_threads',
    'node:test', 'node:fs/promises', 'node:stream/promises',
}

# File type specialization - different file types get different reviewers
FILE_TYPE_REVIEWERS = {
    "config": {
        "patterns": [".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".config.js", ".config.ts", "config.py"],
        "name_patterns": ["config", "settings", ".env", "secrets"],
        "reviewers": ["secrets-scanner", "config-validator"],
        "instructions": {
            "secrets-scanner": """You are a secrets scanner. Check for exposed API keys, passwords, tokens, and sensitive data. Respond with ONLY JSON:
{{"secrets_found": ["secret1"], "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "locations": ["line1"], "recommendation": "text"}}""",
            "config-validator": """You are a configuration validator. Check for misconfigurations, insecure defaults, and missing required settings. Respond with ONLY JSON:
{{"misconfigurations": ["issue1"], "insecure_defaults": ["default1"], "risk_level": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
        }
    },
    "test": {
        "patterns": [".test.js", ".test.ts", ".spec.js", ".spec.ts", "_test.py", "_test.go", "test_"],
        "name_patterns": ["test", "spec", "__tests__"],
        "reviewers": ["test-coverage", "test-quality"],
        "instructions": {
            "test-coverage": """You are a test coverage analyst. Identify missing test cases, edge cases, and untested scenarios. Respond with ONLY JSON:
{{"missing_tests": ["test1"], "edge_cases": ["case1"], "coverage_gaps": ["gap1"], "coverage_rating": "GOOD|FAIR|POOR", "recommendation": "text"}}""",
            "test-quality": """You are a test quality reviewer. Check for test anti-patterns, flaky tests, and assertion quality. Respond with ONLY JSON:
{{"anti_patterns": ["pattern1"], "flaky_indicators": ["indicator1"], "assertion_quality": "GOOD|FAIR|POOR", "recommendation": "text"}}""",
        }
    },
    "entry": {
        "patterns": ["index.js", "index.ts", "main.py", "main.go", "app.py", "app.js", "server.js", "server.ts"],
        "name_patterns": ["index", "main", "app", "server", "entry"],
        "reviewers": ["api-surface", "export-analyzer"],
        "instructions": {
            "api-surface": """You are an API surface analyzer. Identify public APIs, endpoints, and exposed interfaces. Respond with ONLY JSON:
{{"public_apis": ["api1"], "endpoints": ["endpoint1"], "exposure_risk": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
            "export-analyzer": """You are an export analyzer. Check what's being exported, potential leaks, and module boundaries. Respond with ONLY JSON:
{{"exports": ["export1"], "potential_leaks": ["leak1"], "boundary_issues": ["issue1"], "risk_level": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
        }
    },
    "database": {
        "patterns": ["model.py", "models.py", "schema.py", "migration", "db.js", "database.js", "orm"],
        "name_patterns": ["model", "schema", "migration", "database", "db", "repository"],
        "reviewers": ["injection-checker", "data-validation"],
        "instructions": {
            "injection-checker": """You are an injection vulnerability specialist. Check for SQL injection, NoSQL injection, and ORM misuse. Respond with ONLY JSON:
{{"injection_risks": ["risk1"], "unsafe_queries": ["query1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
            "data-validation": """You are a data validation specialist. Check for missing validation, type coercion issues, and data sanitization. Respond with ONLY JSON:
{{"validation_gaps": ["gap1"], "type_issues": ["issue1"], "sanitization_missing": ["field1"], "risk_level": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
        }
    },
    "auth": {
        "patterns": ["auth.py", "auth.js", "authentication", "authorization", "login", "session", "jwt", "oauth"],
        "name_patterns": ["auth", "login", "session", "token", "credential", "password"],
        "reviewers": ["auth-flow", "session-security"],
        "instructions": {
            "auth-flow": """You are an authentication flow specialist. Check for auth bypasses, weak authentication, and flow vulnerabilities. Respond with ONLY JSON:
{{"auth_bypasses": ["bypass1"], "weak_auth": ["weakness1"], "flow_issues": ["issue1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
            "session-security": """You are a session security specialist. Check for session fixation, insecure cookies, and token vulnerabilities. Respond with ONLY JSON:
{{"session_issues": ["issue1"], "cookie_problems": ["problem1"], "token_vulnerabilities": ["vuln1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
        }
    },
}


def get_file_type_category(file_path: str) -> str | None:
    """Determine the file type category for specialized review."""
    file_name = os.path.basename(file_path).lower()
    file_lower = file_path.lower()

    for category, config in FILE_TYPE_REVIEWERS.items():
        for pattern in config.get("patterns", []):
            if file_lower.endswith(pattern) or pattern in file_lower:
                return category
        for pattern in config.get("name_patterns", []):
            if pattern in file_name or pattern in file_lower:
                return category

    return None


def get_specialized_reviewers(file_path: str) -> tuple[list[str], dict[str, str]]:
    """Get specialized reviewers and their instructions for a file type."""
    category = get_file_type_category(file_path)
    if not category:
        return [], {}

    config = FILE_TYPE_REVIEWERS.get(category, {})
    return config.get("reviewers", []), config.get("instructions", {})


# Global state for tracking the agent graph and reports
graph_state = {
    "nodes": [],
    "edges": [],
    "reports": {},
    "transcripts": {},
    "activity": [],
    "status": "idle",
}
graph_lock = asyncio.Lock()
MAX_ACTIVITY_ITEMS = 50


async def add_node(node_id: str, name: str, node_type: str, parent_id: Optional[str] = None, meta: dict = None):
    """Add a node to the graph."""
    async with graph_lock:
        node = {
            "id": node_id,
            "name": name,
            "type": node_type,
            "status": "pending",
            "parent": parent_id,
            "meta": meta or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        graph_state["nodes"].append(node)
        if parent_id:
            graph_state["edges"].append({"source": parent_id, "target": node_id})
        return node


async def add_edge(source_id: str, target_id: str):
    """Add an edge between two nodes."""
    async with graph_lock:
        graph_state["edges"].append({"source": source_id, "target": target_id})


async def update_node_status(node_id: str, status: str, report: dict = None):
    """Update a node's status and optionally add a report."""
    async with graph_lock:
        for node in graph_state["nodes"]:
            if node["id"] == node_id:
                node["status"] = status
                node["updated_at"] = datetime.utcnow().isoformat()
                break
        if report:
            graph_state["reports"][node_id] = report


async def add_transcript_entry(node_id: str, role: str, content: str, entry_type: str = "message"):
    """Add an entry to a node's transcript."""
    async with graph_lock:
        if node_id not in graph_state["transcripts"]:
            graph_state["transcripts"][node_id] = []

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content,
            "type": entry_type,
        }
        graph_state["transcripts"][node_id].append(entry)

        node_name = node_id
        for node in graph_state["nodes"]:
            if node["id"] == node_id:
                node_name = node.get("name", node_id)
                break

        activity_entry = {
            "timestamp": entry["timestamp"],
            "node_id": node_id,
            "node_name": node_name,
            "role": role,
            "content": content[:200] + ("..." if len(content) > 200 else ""),
            "type": entry_type,
        }
        graph_state["activity"].insert(0, activity_entry)

        if len(graph_state["activity"]) > MAX_ACTIVITY_ITEMS:
            graph_state["activity"] = graph_state["activity"][:MAX_ACTIVITY_ITEMS]


async def reset_graph():
    """Reset the graph state."""
    async with graph_lock:
        graph_state["nodes"] = []
        graph_state["edges"] = []
        graph_state["reports"] = {}
        graph_state["transcripts"] = {}
        graph_state["activity"] = []
        graph_state["status"] = "idle"


# ============== GitHub / Registry Helpers ==============

def get_npm_info(package_name: str) -> dict:
    """Fetch package info from npm registry."""
    try:
        url = f"https://registry.npmjs.org/{package_name}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


def get_npm_package_files(package_name: str, max_files: int = 5) -> list[dict]:
    """Fetch key source files from an npm package via its GitHub repository."""
    try:
        info = get_npm_info(package_name)
        if "error" in info:
            return [{"error": info["error"]}]

        repo = info.get("repository", {})
        if isinstance(repo, str):
            repo_url = repo
        elif isinstance(repo, dict):
            repo_url = repo.get("url", "")
        else:
            return [{"error": "No repository URL found"}]

        if not repo_url:
            return [{"error": "No repository URL found"}]

        github_match = re.search(r'github\.com[/:]([^/]+)/([^/.\s#]+)', repo_url)
        if not github_match:
            github_match = re.search(r'^github:([^/]+)/([^/.\s#]+)', repo_url)

        if not github_match:
            return [{"error": f"Could not parse GitHub URL: {repo_url}"}]

        org, repo_name = github_match.groups()
        repo_name = repo_name.rstrip('.git')

        files = get_github_files(org, repo_name, "main", NODE_EXTENSIONS, max_files)
        if files and not (len(files) == 1 and "error" in files[0]):
            return files

        files = get_github_files(org, repo_name, "master", NODE_EXTENSIONS, max_files)
        if files and not (len(files) == 1 and "error" in files[0]):
            return files

        return [{"error": "Could not fetch files from repository"}]
    except Exception as e:
        return [{"error": str(e)}]


def get_pypi_package_files(package_name: str, max_files: int = 5) -> list[dict]:
    """Fetch key source files from a PyPI package."""
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        project_urls = data.get("info", {}).get("project_urls") or {}
        home_page = data.get("info", {}).get("home_page", "")

        github_url = None
        for key, val in project_urls.items():
            if val and "github.com" in val.lower():
                github_url = val
                break

        if not github_url and home_page and "github.com" in home_page.lower():
            github_url = home_page

        if github_url:
            match = re.search(r'github\.com[/:]([^/]+)/([^/.\s]+)', github_url)
            if match:
                org, repo = match.groups()
                repo = repo.rstrip('.git')
                files = get_github_files(org, repo, "main", [".py"], max_files)
                if files and not (len(files) == 1 and "error" in files[0]):
                    return files
                files = get_github_files(org, repo, "master", [".py"], max_files)
                if files and not (len(files) == 1 and "error" in files[0]):
                    return files

        return []
    except Exception as e:
        return [{"error": str(e)}]


def get_github_files(org: str, repo: str, branch: str = "main", extensions: list = None, max_files: int = 50) -> list[dict]:
    """Fetch files from a GitHub repository."""
    if extensions is None:
        extensions = PYTHON_EXTENSIONS

    try:
        url = f"https://github.com/{org}/{repo}/archive/refs/heads/{branch}.zip"
        zip_path = f"/tmp/{repo}-{branch}.zip"
        urllib.request.urlretrieve(url, zip_path)

        extracted = f"/tmp/{repo}-{branch}-extracted"
        if os.path.exists(extracted):
            shutil.rmtree(extracted)

        os.makedirs(extracted)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted)

        os.remove(zip_path)

        files = []
        for root, dirs, file_list in os.walk(extracted):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]

            for file in file_list:
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    stat = os.stat(full_path)
                    if stat.st_size < 50 * 1024:
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            relative_path = os.path.relpath(full_path, extracted)
                            parts = relative_path.split(os.sep)
                            if len(parts) > 1:
                                relative_path = os.sep.join(parts[1:])
                            files.append({
                                "path": relative_path,
                                "content": content,
                                "size": stat.st_size,
                                "lines": len(content.splitlines())
                            })
                        except:
                            pass

                        if max_files > 0 and len(files) >= max_files:
                            break
            if max_files > 0 and len(files) >= max_files:
                break

        shutil.rmtree(extracted)
        return files
    except Exception as e:
        return [{"error": str(e)}]


def detect_repo_language(org: str, repo: str, branch: str = None) -> tuple[str, str]:
    """Detect project language and branch by checking for manifest files."""
    branches_to_try = [branch] if branch else []
    branches_to_try.extend(["main", "master"])
    branches_to_try = list(dict.fromkeys(branches_to_try))

    for try_branch in branches_to_try:
        url = f"https://raw.githubusercontent.com/{org}/{repo}/{try_branch}/package.json"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return (LANGUAGE_NODE, try_branch)
        except:
            pass

        url = f"https://raw.githubusercontent.com/{org}/{repo}/{try_branch}/pyproject.toml"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return (LANGUAGE_PYTHON, try_branch)
        except:
            pass

        url = f"https://raw.githubusercontent.com/{org}/{repo}/{try_branch}/requirements.txt"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return (LANGUAGE_PYTHON, try_branch)
        except:
            pass

        url = f"https://raw.githubusercontent.com/{org}/{repo}/{try_branch}/setup.py"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return (LANGUAGE_PYTHON, try_branch)
        except:
            pass

    return (LANGUAGE_PYTHON, branch or "main")


def extensions_for_language(language: str) -> list:
    """Return file extensions for the given language."""
    if language == LANGUAGE_NODE:
        return NODE_EXTENSIONS
    return PYTHON_EXTENSIONS


def get_file_language(file_path: str, default_language: str = LANGUAGE_PYTHON) -> str:
    """Determine the language of a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in NODE_EXTENSIONS:
        return LANGUAGE_NODE
    if ext in PYTHON_EXTENSIONS:
        return LANGUAGE_PYTHON
    return default_language


# ============== Import Extraction ==============

def normalize_js_module(module: str) -> str:
    module = module.strip()
    if module.startswith((".", "/")):
        return ""
    if module.startswith("@"):
        parts = module.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else module
    return module.split("/")[0]


def extract_imports_from_js(code: str) -> set[str]:
    imports = set()
    patterns = [
        r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"import\s+(?:[^;]+?\s+from\s+)?['\"]([^'\"]+)['\"]",
        r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"export\s+\*\s+from\s+['\"]([^'\"]+)['\"]",
        r"export\s+\{[^}]+\}\s+from\s+['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            module = normalize_js_module(match.group(1))
            if module:
                imports.add(module)
    return imports - NODE_BUILTINS


def extract_imports_from_code(code: str, language: str = "python") -> set[str]:
    """Extract import statements from code and return package names."""
    lang = (language or "python").lower()
    if lang in ("node", "javascript", "typescript", "js", "ts"):
        return extract_imports_from_js(code)

    imports = set()

    for match in re.finditer(r'^import\s+([\w.]+)', code, re.MULTILINE):
        pkg = match.group(1).split('.')[0]
        imports.add(pkg)

    for match in re.finditer(r'^from\s+([\w.]+)\s+import', code, re.MULTILINE):
        pkg = match.group(1).split('.')[0]
        imports.add(pkg)

    stdlib = {
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
        'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools',
        'glob', 'hashlib', 'heapq', 'html', 'http', 'importlib', 'inspect', 'io',
        'itertools', 'json', 'logging', 'math', 'multiprocessing', 'operator', 'os',
        'pathlib', 'pickle', 'platform', 'pprint', 'queue', 'random', 're', 'shutil',
        'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'struct', 'subprocess',
        'sys', 'tempfile', 'textwrap', 'threading', 'time', 'traceback', 'types',
        'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile',
        '__future__', 'builtins', 'typing_extensions',
    }

    return imports - stdlib


# Dependency data cache
dep_cache = {}

# Track discovered dependencies during repo analysis
discovered_deps = set()
discovered_deps_lock = asyncio.Lock()


async def fetch_dependency_data(dep_name: str, language: str = LANGUAGE_PYTHON) -> dict:
    """Fetch data about a dependency for reviewers to analyze."""
    cache_key = f"{language}:{dep_name}"
    if cache_key in dep_cache:
        return dep_cache[cache_key]

    if language == LANGUAGE_NODE:
        info = await asyncio.to_thread(get_npm_info, dep_name)

        if "error" in info:
            data = {"name": dep_name, "error": info["error"], "language": language}
        else:
            latest = info.get("dist-tags", {}).get("latest", "unknown")
            version_info = info.get("versions", {}).get(latest, {}) if latest != "unknown" else {}
            data = {
                "name": dep_name,
                "version": latest,
                "summary": info.get("description", ""),
                "author": info.get("author", {}).get("name", "unknown") if isinstance(info.get("author"), dict) else info.get("author", "unknown"),
                "license": version_info.get("license", info.get("license", "unknown")),
                "home_page": info.get("homepage", ""),
                "repository": info.get("repository", {}).get("url", "") if isinstance(info.get("repository"), dict) else info.get("repository", ""),
                "keywords": info.get("keywords", []),
                "language": language,
            }
    else:
        try:
            url = f"https://pypi.org/pypi/{dep_name}/json"
            with urllib.request.urlopen(url, timeout=10) as response:
                info = json.loads(response.read().decode())
        except Exception as e:
            info = {"error": str(e)}

        if "error" in info:
            data = {"name": dep_name, "error": info["error"], "language": language}
        else:
            pkg_info = info.get("info", {})
            data = {
                "name": dep_name,
                "version": pkg_info.get("version", "unknown"),
                "summary": pkg_info.get("summary", ""),
                "author": pkg_info.get("author", "unknown"),
                "license": pkg_info.get("license", "unknown"),
                "home_page": pkg_info.get("home_page", ""),
                "project_url": pkg_info.get("project_url", ""),
                "requires_python": pkg_info.get("requires_python", ""),
                "language": language,
            }

    dep_cache[cache_key] = data
    return data


# ============== Distributed File Review Worker ==============

class FileReviewWorker:
    """Worker that runs on remote runners to review files."""

    async def review_single_file(self, file_info: dict, reviewers: list, language: str) -> dict:
        """Review a single file with multiple reviewer agents."""
        from autonomy import Agent, Model
        import json
        import asyncio

        file_path = file_info["path"]
        file_content = file_info["content"]

        max_content_len = 8000
        truncated = file_content[:max_content_len] if len(file_content) > max_content_len else file_content

        instructions = {
            "security": f"""Analyze this code file for security issues. Respond with ONLY JSON:
{{"risk": "LOW|MEDIUM|HIGH|CRITICAL", "issues": ["issue1"], "recommendation": "text"}}""",

            "quality": f"""Analyze this code file for code quality. Respond with ONLY JSON:
{{"quality": "GOOD|FAIR|POOR", "issues": ["issue1"], "recommendation": "text"}}""",

            "complexity": f"""Analyze this code file for complexity. Respond with ONLY JSON:
{{"complexity": "LOW|MEDIUM|HIGH", "metrics": ["metric1"], "recommendation": "text"}}""",

            "documentation": f"""Analyze this code file for documentation quality. Respond with ONLY JSON:
{{"doc_quality": "GOOD|FAIR|POOR", "missing": ["item1"], "recommendation": "text"}}""",
        }

        model = Model(
            "nova-micro-v1",
            throttle=True,
            throttle_requests_per_minute=1000,
            throttle_max_requests_in_progress=100,
            request_timeout=300.0,
        )

        results = {}

        for reviewer_type in reviewers:
            agent = await Agent.start(
                node=self.node,
                instructions=instructions.get(reviewer_type, "Review this code."),
                model=model,
                max_execution_time=900.0,
                max_iterations=3,
            )

            try:
                message = f"Review this file: {file_path}\n\n```\n{truncated}\n```"
                response = await agent.send(message, timeout=660)

                if response and len(response) > 0:
                    content = response[-1].content
                    result_text = content.text if hasattr(content, 'text') else str(content)
                else:
                    result_text = "No response"

                try:
                    first_brace = result_text.find('{')
                    last_brace = result_text.rfind('}')
                    if first_brace != -1 and last_brace != -1:
                        result = json.loads(result_text[first_brace:last_brace + 1])
                    else:
                        result = {"raw_response": result_text}
                except:
                    result = {"raw_response": result_text}

                result["reviewer_type"] = reviewer_type
                result["file"] = file_path
                results[reviewer_type] = result

            except Exception as e:
                results[reviewer_type] = {"error": str(e), "reviewer_type": reviewer_type, "file": file_path}
            finally:
                asyncio.create_task(Agent.stop(self.node, agent.name, timeout=5.0))

        return {
            "file": file_path,
            "reviews": results,
            "detected_imports": [],
            "lines": file_info.get("lines", 0),
        }

    async def handle_message(self, context, message):
        """Handle a batch of files to review with progress updates."""
        import json
        import os
        import asyncio

        request = json.loads(message)
        files = request.get("files", [])
        reviewers = request.get("reviewers", ["security", "quality"])
        language = request.get("language", "python")
        runner_id = request.get("runner_id", "unknown")

        results = []
        total_files = len(files)
        completed_count = 0
        results_lock = asyncio.Lock()

        PARALLEL_FILES = 100
        semaphore = asyncio.Semaphore(PARALLEL_FILES)

        async def process_file(idx, file_info):
            nonlocal completed_count
            async with semaphore:
                file_path = file_info.get("path", "unknown")
                short_name = os.path.basename(file_path)
                dir_path = os.path.dirname(file_path)

                progress = json.dumps({
                    "type": "progress",
                    "runner_id": runner_id,
                    "status": "starting",
                    "file": short_name,
                    "file_path": file_path,
                    "dir_path": dir_path,
                    "file_index": idx + 1,
                    "total_files": total_files
                })
                await context.reply(progress)

                try:
                    result = await self.review_single_file(file_info, reviewers, language)

                    async with results_lock:
                        results.append(result)
                        completed_count += 1

                    progress = json.dumps({
                        "type": "progress",
                        "runner_id": runner_id,
                        "status": "completed",
                        "file": short_name,
                        "file_path": file_path,
                        "dir_path": dir_path,
                        "file_index": idx + 1,
                        "total_files": total_files,
                        "completed_count": completed_count,
                        "result": result
                    })
                    await context.reply(progress)

                except Exception as e:
                    async with results_lock:
                        results.append({"file": file_path, "error": str(e)})
                        completed_count += 1

                    progress = json.dumps({
                        "type": "progress",
                        "runner_id": runner_id,
                        "status": "error",
                        "file": short_name,
                        "file_path": file_path,
                        "dir_path": dir_path,
                        "file_index": idx + 1,
                        "total_files": total_files,
                        "completed_count": completed_count,
                        "error": str(e)
                    })
                    await context.reply(progress)

        tasks = [process_file(idx, file_info) for idx, file_info in enumerate(files)]
        await asyncio.gather(*tasks)

        final = json.dumps({
            "type": "final",
            "runner_id": runner_id,
            "results": results
        })
        await context.reply(final)


def split_list_into_n_parts(lst, n):
    """Split a list into n roughly equal parts."""
    if n <= 0:
        return [lst]
    q, r = divmod(len(lst), n)
    return [lst[i * q + min(i, r): (i + 1) * q + min(i + 1, r)] for i in range(n)]


async def run_distributed_file_review(node, files: list, reviewers: list, language: str, root_id: str = None) -> list:
    """Distribute file reviews across available runner nodes."""
    import secrets

    runners = await Zone.nodes(node, filter="runner")
    num_runners = len(runners)

    if num_runners == 0:
        return None

    if root_id:
        await add_transcript_entry(root_id, "system", f"Found {num_runners} runner nodes", "message")

    if num_runners > len(files):
        runners = runners[:len(files)]
        num_runners = len(runners)

    file_batches = split_list_into_n_parts(files, num_runners)

    if root_id:
        await add_transcript_entry(root_id, "system", f"Distributing {len(files)} files across {num_runners} runners (~{len(files)//num_runners} files each)", "message")

    runner_node_ids = []
    for i in range(num_runners):
        runner_node_id = f"runner-{i+1}-{secrets.token_hex(3)}"
        await add_node(runner_node_id, f"Runner {i+1}", "runner", root_id, {
            "runner_index": i+1,
            "file_count": len(file_batches[i]),
            "runner_name": runners[i].name
        })
        await update_node_status(runner_node_id, "running")
        runner_node_ids.append(runner_node_id)

    runner_file_nodes = [{} for _ in range(num_runners)]
    runner_dir_nodes = [{} for _ in range(num_runners)]

    async def run_on_runner(runner_idx, runner, batch):
        worker_name = f"reviewer-{secrets.token_hex(3)}"
        runner_id = f"Runner {runner_idx + 1}"
        runner_node_id = runner_node_ids[runner_idx]

        if root_id:
            await add_transcript_entry(root_id, "system", f"[{runner_id}] Starting worker with {len(batch)} files...", "message")

        await runner.start_worker(worker_name, FileReviewWorker())

        request = json.dumps({
            "files": batch,
            "reviewers": reviewers,
            "language": language,
            "runner_id": runner_id,
        })

        results = []
        try:
            mailbox = await node.send(worker_name, request, node=runner.name)

            while True:
                try:
                    reply_json = await mailbox.receive(timeout=1800)
                    msg = json.loads(reply_json)
                    msg_type = msg.get("type", "")

                    if msg_type == "progress":
                        status = msg.get("status", "")
                        file_name = msg.get("file", "")
                        file_path = msg.get("file_path", file_name)
                        dir_path = msg.get("dir_path", "")
                        file_idx = msg.get("file_index", 0)
                        total = msg.get("total_files", 0)

                        if status == "starting":
                            if file_path in runner_file_nodes[runner_idx]:
                                continue

                            parent_node_id = runner_node_id
                            if dir_path:
                                if dir_path not in runner_dir_nodes[runner_idx]:
                                    dir_node_id = f"dir-{runner_idx}-{secrets.token_hex(3)}"
                                    dir_name = dir_path.split("/")[-1] if "/" in dir_path else dir_path
                                    await add_node(dir_node_id, dir_name, "directory", runner_node_id, {
                                        "path": dir_path,
                                        "runner": runner_id
                                    })
                                    await update_node_status(dir_node_id, "running")
                                    runner_dir_nodes[runner_idx][dir_path] = dir_node_id
                                parent_node_id = runner_dir_nodes[runner_idx][dir_path]

                            file_node_id = f"file-{runner_idx}-{file_idx}-{secrets.token_hex(3)}"
                            await add_node(file_node_id, file_name, "file", parent_node_id, {
                                "path": file_path,
                                "runner": runner_id
                            })
                            await update_node_status(file_node_id, "running")
                            runner_file_nodes[runner_idx][file_path] = file_node_id

                            if root_id:
                                await add_transcript_entry(root_id, "system", f"[{runner_id}] {file_name}", "message")

                        elif status == "completed":
                            file_node_id = runner_file_nodes[runner_idx].get(file_path)
                            if file_node_id:
                                result_data = msg.get("result", {})
                                await update_node_status(file_node_id, "completed", result_data)

                                reviews = result_data.get("reviews", {})
                                for reviewer_type, review_result in reviews.items():
                                    reviewer_node_id = f"reviewer-{runner_idx}-{file_idx}-{reviewer_type}-{secrets.token_hex(2)}"
                                    await add_node(reviewer_node_id, reviewer_type.title(), "reviewer", file_node_id, {
                                        "reviewer_type": reviewer_type,
                                        "file": file_name,
                                        "runner": runner_id
                                    })
                                    await update_node_status(reviewer_node_id, "completed", review_result)

                            completed = msg.get("completed_count", file_idx)
                            if root_id:
                                await add_transcript_entry(root_id, "system", f"[{runner_id}] Done: {file_name} ({completed}/{total})", "message")

                        elif status == "error":
                            file_node_id = runner_file_nodes[runner_idx].get(file_path)
                            if file_node_id:
                                await update_node_status(file_node_id, "error", {"error": msg.get("error", "unknown")})

                            if root_id:
                                await add_transcript_entry(root_id, "error", f"[{runner_id}] Error: {file_name} - {msg.get('error', 'unknown')}", "error")

                    elif msg_type == "final":
                        results = msg.get("results", [])
                        for dir_node_id in runner_dir_nodes[runner_idx].values():
                            await update_node_status(dir_node_id, "completed")
                        await update_node_status(runner_node_id, "completed")
                        if root_id:
                            await add_transcript_entry(root_id, "system", f"[{runner_id}] Finished all {len(batch)} files", "message")
                        break

                except Exception as e:
                    if root_id:
                        await add_transcript_entry(root_id, "error", f"[{runner_id}] Receive error: {str(e)}", "error")
                    break

        except Exception as e:
            results = [{"error": str(e)}]
            await update_node_status(runner_node_id, "error")
            if root_id:
                await add_transcript_entry(root_id, "error", f"[{runner_id}] Error: {str(e)}", "error")
        finally:
            try:
                await runner.stop_worker(worker_name)
            except Exception:
                pass

        return results

    futures = [run_on_runner(i, runners[i], file_batches[i]) for i in range(num_runners)]
    batch_results = await asyncio.gather(*futures)

    all_results = []
    for batch in batch_results:
        if isinstance(batch, list):
            all_results.extend(batch)

    return all_results


# ============== Sub-Reviewers (Adaptive Deep Dive) ==============

SUB_REVIEWERS = {
    "security": {
        "high_risk": ["vulnerability-scanner", "input-validation", "auth-checker"],
        "instructions": {
            "vulnerability-scanner": """You are a vulnerability scanner. Deep-dive into potential CVEs and known vulnerability patterns. Respond with ONLY JSON:
{{"vulnerabilities": ["vuln1"], "cve_patterns": ["pattern1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
            "input-validation": """You are an input validation specialist. Check for injection vulnerabilities, unsanitized inputs, and data validation issues. Respond with ONLY JSON:
{{"injection_risks": ["risk1"], "validation_gaps": ["gap1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
            "auth-checker": """You are an authentication/authorization specialist. Check for auth bypasses, weak auth patterns, and permission issues. Respond with ONLY JSON:
{{"auth_issues": ["issue1"], "permission_gaps": ["gap1"], "severity": "LOW|MEDIUM|HIGH|CRITICAL", "recommendation": "text"}}""",
        }
    },
    "complexity": {
        "high_risk": ["refactoring-advisor", "test-coverage-analyzer"],
        "instructions": {
            "refactoring-advisor": """You are a refactoring specialist. Identify code that should be refactored and suggest improvements. Respond with ONLY JSON:
{{"refactor_targets": ["target1"], "suggested_patterns": ["pattern1"], "priority": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
            "test-coverage-analyzer": """You are a test coverage analyst. Identify untested code paths and suggest test cases. Respond with ONLY JSON:
{{"untested_paths": ["path1"], "suggested_tests": ["test1"], "coverage_risk": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
        }
    },
    "quality": {
        "high_risk": ["code-smell-detector", "maintainability-reviewer"],
        "instructions": {
            "code-smell-detector": """You are a code smell detector. Identify anti-patterns, dead code, and code smells. Respond with ONLY JSON:
{{"code_smells": ["smell1"], "anti_patterns": ["pattern1"], "severity": "LOW|MEDIUM|HIGH", "recommendation": "text"}}""",
            "maintainability-reviewer": """You are a maintainability specialist. Assess long-term maintenance burden and technical debt. Respond with ONLY JSON:
{{"tech_debt": ["debt1"], "maintenance_risks": ["risk1"], "maintainability": "GOOD|FAIR|POOR", "recommendation": "text"}}""",
        }
    },
    "documentation": {
        "high_risk": ["api-doc-reviewer", "example-checker"],
        "instructions": {
            "api-doc-reviewer": """You are an API documentation specialist. Check for missing or unclear API documentation. Respond with ONLY JSON:
{{"missing_docs": ["doc1"], "unclear_apis": ["api1"], "doc_quality": "GOOD|FAIR|POOR", "recommendation": "text"}}""",
            "example-checker": """You are a code example specialist. Check for missing usage examples and unclear code. Respond with ONLY JSON:
{{"missing_examples": ["example1"], "unclear_code": ["code1"], "clarity": "GOOD|FAIR|POOR", "recommendation": "text"}}""",
        }
    }
}


def should_spawn_sub_reviewers(reviewer_type: str, result: dict) -> bool:
    """Determine if a reviewer's findings warrant spawning sub-reviewers."""
    if reviewer_type == "security":
        return result.get("risk", "").upper() in ["HIGH", "CRITICAL"]
    elif reviewer_type == "complexity":
        return result.get("complexity", "").upper() == "HIGH"
    elif reviewer_type == "quality":
        return result.get("quality", "").upper() == "POOR"
    elif reviewer_type == "documentation":
        return result.get("doc_quality", "").upper() == "POOR"
    return False


async def run_sub_reviewer_agent(node, node_id: str, sub_reviewer_type: str, parent_reviewer_type: str, file_path: str, file_content: str, parent_findings: dict) -> dict:
    """Run a specialized sub-reviewer agent based on parent reviewer findings."""

    await update_node_status(node_id, "running")
    await add_transcript_entry(node_id, "system", f"Starting {sub_reviewer_type} deep-dive for: {file_path}", "message")

    instructions = SUB_REVIEWERS.get(parent_reviewer_type, {}).get("instructions", {}).get(sub_reviewer_type, "Analyze this code.")

    max_content_len = 8000
    truncated = file_content[:max_content_len] if len(file_content) > max_content_len else file_content

    model = Model(
        "nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=1000,
        throttle_max_requests_in_progress=100,
        request_timeout=300.0,
    )

    agent = await Agent.start(
        node=node,
        instructions=instructions,
        model=model,
        max_execution_time=900.0,
        max_iterations=3,
    )

    try:
        parent_context = json.dumps(parent_findings, indent=2)[:1000]
        message = f"Deep-dive analysis of: {file_path}\n\nParent reviewer ({parent_reviewer_type}) found these issues:\n{parent_context}\n\nCode:\n```\n{truncated}\n```"

        await add_transcript_entry(node_id, "agent", f"Deep-diving {file_path}", "message")

        response = await agent.send(message, timeout=660)

        if response and len(response) > 0:
            content = response[-1].content
            result_text = content.text if hasattr(content, 'text') else str(content)
        else:
            result_text = "No response"

        await add_transcript_entry(node_id, "agent", result_text, "message")

        try:
            first_brace = result_text.find('{')
            last_brace = result_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                result = json.loads(result_text[first_brace:last_brace + 1])
            else:
                result = {"raw_response": result_text}
        except:
            result = {"raw_response": result_text}

        result["sub_reviewer_type"] = sub_reviewer_type
        result["parent_reviewer"] = parent_reviewer_type
        result["file"] = file_path
        await update_node_status(node_id, "completed", result)
        return result

    except Exception as e:
        await add_transcript_entry(node_id, "error", str(e), "error")
        await update_node_status(node_id, "error", {"error": str(e)})
        return {"error": str(e)}
    finally:
        asyncio.create_task(Agent.stop(node, agent.name, timeout=5.0))


async def spawn_sub_reviewers(node, reviewer_type: str, result: dict, file_node_id: str, file_path: str, file_content: str) -> list:
    """Spawn sub-reviewers based on a reviewer's concerning findings."""

    sub_reviewer_config = SUB_REVIEWERS.get(reviewer_type, {})
    sub_reviewer_types = sub_reviewer_config.get("high_risk", [])

    if not sub_reviewer_types:
        return []

    await add_transcript_entry(file_node_id, "system", f"{reviewer_type} found concerning issues - spawning {len(sub_reviewer_types)} sub-reviewers", "message")

    tasks = []
    for sub_type in sub_reviewer_types:
        sub_node_id = f"sub-{sub_type}-{uuid.uuid4().hex[:6]}"
        await add_node(sub_node_id, sub_type.replace("-", " ").title(), "sub-reviewer", file_node_id, {"sub_reviewer_type": sub_type, "parent_reviewer": reviewer_type})

        task = run_sub_reviewer_agent(node, sub_node_id, sub_type, reviewer_type, file_path, file_content, result)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


async def run_specialized_reviewer_agent(node, node_id: str, reviewer_type: str, file_path: str, file_content: str, instructions: str) -> dict:
    """Run a specialized reviewer agent for a specific file type."""

    await update_node_status(node_id, "running")
    await add_transcript_entry(node_id, "system", f"Starting specialized {reviewer_type} review for: {file_path}", "message")

    max_content_len = 8000
    truncated = file_content[:max_content_len] if len(file_content) > max_content_len else file_content

    model = Model(
        "nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=1000,
        throttle_max_requests_in_progress=100,
        request_timeout=300.0,
    )

    agent = await Agent.start(
        node=node,
        instructions=instructions,
        model=model,
        max_execution_time=900.0,
        max_iterations=3,
    )

    try:
        message = f"Specialized review of: {file_path}\n\n```\n{truncated}\n```"

        await add_transcript_entry(node_id, "agent", f"Specialized review of {file_path}", "message")

        response = await agent.send(message, timeout=660)

        if response and len(response) > 0:
            content = response[-1].content
            result_text = content.text if hasattr(content, 'text') else str(content)
        else:
            result_text = "No response"

        await add_transcript_entry(node_id, "agent", result_text, "message")

        try:
            first_brace = result_text.find('{')
            last_brace = result_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                result = json.loads(result_text[first_brace:last_brace + 1])
            else:
                result = {"raw_response": result_text}
        except:
            result = {"raw_response": result_text}

        result["specialized_reviewer_type"] = reviewer_type
        result["file"] = file_path
        await update_node_status(node_id, "completed", result)
        return result

    except Exception as e:
        await add_transcript_entry(node_id, "error", str(e), "error")
        await update_node_status(node_id, "error", {"error": str(e)})
        return {"error": str(e)}
    finally:
        asyncio.create_task(Agent.stop(node, agent.name, timeout=5.0))


async def spawn_specialized_reviewers(node, file_node_id: str, file_path: str, file_content: str) -> list:
    """Spawn specialized reviewers based on file type."""

    reviewer_types, instructions = get_specialized_reviewers(file_path)

    if not reviewer_types:
        return []

    category = get_file_type_category(file_path)
    await add_transcript_entry(file_node_id, "system", f"File type '{category}' detected - spawning {len(reviewer_types)} specialized reviewers", "message")

    tasks = []
    for reviewer_type in reviewer_types:
        spec_node_id = f"spec-{reviewer_type}-{uuid.uuid4().hex[:6]}"
        await add_node(spec_node_id, reviewer_type.replace("-", " ").title(), "specialized", file_node_id, {"specialized_type": reviewer_type, "file_category": category})

        reviewer_instructions = instructions.get(reviewer_type, "Analyze this code.")
        task = run_specialized_reviewer_agent(node, spec_node_id, reviewer_type, file_path, file_content, reviewer_instructions)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


# ============== File Review ==============

async def run_file_reviewer_agent(node, node_id: str, reviewer_type: str, file_path: str, file_content: str, extract_imports: bool = False, language: str = LANGUAGE_PYTHON) -> dict:
    """Run a reviewer agent for a single file."""

    await update_node_status(node_id, "running")
    await add_transcript_entry(node_id, "system", f"Starting {reviewer_type} review for: {file_path}", "message")

    detected_imports = set()
    if extract_imports:
        detected_imports = extract_imports_from_code(file_content, language)
        if detected_imports:
            await add_transcript_entry(node_id, "system", f"Detected imports: {', '.join(sorted(detected_imports))}", "message")

    max_content_len = 8000
    truncated = file_content[:max_content_len] if len(file_content) > max_content_len else file_content

    instructions = {
        "security": f"""Analyze this code file for security issues. Respond with ONLY JSON:
{{"risk": "LOW|MEDIUM|HIGH|CRITICAL", "issues": ["issue1"], "recommendation": "text"}}""",

        "quality": f"""Analyze this code file for code quality. Respond with ONLY JSON:
{{"quality": "GOOD|FAIR|POOR", "issues": ["issue1"], "recommendation": "text"}}""",

        "complexity": f"""Analyze this code file for complexity. Respond with ONLY JSON:
{{"complexity": "LOW|MEDIUM|HIGH", "metrics": ["metric1"], "recommendation": "text"}}""",

        "documentation": f"""Analyze this code file for documentation quality. Respond with ONLY JSON:
{{"doc_quality": "GOOD|FAIR|POOR", "missing": ["item1"], "recommendation": "text"}}""",
    }

    model = Model(
        "nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=1000,
        throttle_max_requests_in_progress=100,
        request_timeout=300.0,
    )

    agent = await Agent.start(
        node=node,
        instructions=instructions.get(reviewer_type, "Review this code."),
        model=model,
        max_execution_time=900.0,
        max_iterations=3,
    )

    try:
        message = f"Review this file: {file_path}\n\n```\n{truncated}\n```"
        await add_transcript_entry(node_id, "agent", f"Reviewing {file_path} ({len(file_content)} chars)", "message")

        response = await agent.send(message, timeout=660)

        if response and len(response) > 0:
            content = response[-1].content
            result_text = content.text if hasattr(content, 'text') else str(content)
        else:
            result_text = "No response"

        await add_transcript_entry(node_id, "agent", result_text, "message")

        try:
            first_brace = result_text.find('{')
            last_brace = result_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                result = json.loads(result_text[first_brace:last_brace + 1])
            else:
                result = {"raw_response": result_text}
        except:
            result = {"raw_response": result_text}

        result["reviewer_type"] = reviewer_type
        result["file"] = file_path
        result["detected_imports"] = list(detected_imports) if detected_imports else []
        await update_node_status(node_id, "completed", result)
        return result

    except Exception as e:
        await add_transcript_entry(node_id, "error", str(e), "error")
        await update_node_status(node_id, "error", {"error": str(e)})
        return {"error": str(e), "detected_imports": list(detected_imports) if detected_imports else []}
    finally:
        asyncio.create_task(Agent.stop(node, agent.name, timeout=5.0))


async def review_file(node, file_info: dict, parent_id: str, reviewers: list = None, analyze_deps: bool = False, root_id: str = None, language: str = LANGUAGE_PYTHON, dep_depth: int = 0, max_dep_depth: int = 2):
    """Review a single file with multiple reviewer agents."""

    if reviewers is None:
        reviewers = ["security", "quality"]

    file_path = file_info["path"]
    file_content = file_info["content"]

    file_language = get_file_language(file_path, language)

    file_node_id = f"file-{uuid.uuid4().hex[:8]}"
    short_name = os.path.basename(file_path)
    await add_node(file_node_id, short_name, "file", parent_id, {"path": file_path, "lines": file_info.get("lines", 0), "language": file_language})
    await update_node_status(file_node_id, "running")
    await add_transcript_entry(file_node_id, "system", f"Reviewing file: {file_path} ({file_info.get('lines', '?')} lines)", "message")

    reviewer_tasks = []
    for i, reviewer_type in enumerate(reviewers):
        reviewer_node_id = f"reviewer-{short_name}-{reviewer_type}-{uuid.uuid4().hex[:6]}"
        await add_node(reviewer_node_id, reviewer_type.title(), "reviewer", file_node_id, {"reviewer_type": reviewer_type})

        extract_imports = (i == 0) and analyze_deps
        task = run_file_reviewer_agent(node, reviewer_node_id, reviewer_type, file_path, file_content, extract_imports=extract_imports, language=file_language)
        reviewer_tasks.append(task)

    await add_transcript_entry(file_node_id, "system", f"Spawned {len(reviewer_tasks)} reviewers", "message")

    results = await asyncio.gather(*reviewer_tasks, return_exceptions=True)

    all_imports = set()
    for r in results:
        if isinstance(r, dict) and "detected_imports" in r:
            all_imports.update(r["detected_imports"])

    sub_reviewer_tasks = []
    for result in results:
        if isinstance(result, dict) and "reviewer_type" in result:
            reviewer_type = result["reviewer_type"]
            if should_spawn_sub_reviewers(reviewer_type, result):
                sub_reviewer_tasks.append(
                    spawn_sub_reviewers(node, reviewer_type, result, file_node_id, file_path, file_content)
                )

    sub_results = []
    if sub_reviewer_tasks:
        await add_transcript_entry(file_node_id, "system", f"Spawning sub-reviewers for {len(sub_reviewer_tasks)} concerning findings", "message")
        sub_results_nested = await asyncio.gather(*sub_reviewer_tasks, return_exceptions=True)
        for sr in sub_results_nested:
            if isinstance(sr, list):
                sub_results.extend(sr)

    specialized_results = await spawn_specialized_reviewers(node, file_node_id, file_path, file_content)

    aggregated = {
        "file": file_path,
        "file_category": get_file_type_category(file_path),
        "reviews": {r.get("reviewer_type", "unknown"): r for r in results if isinstance(r, dict)},
        "sub_reviews": {r.get("sub_reviewer_type", "unknown"): r for r in sub_results if isinstance(r, dict)},
        "specialized_reviews": {r.get("specialized_reviewer_type", "unknown"): r for r in specialized_results if isinstance(r, dict)},
        "detected_imports": list(all_imports),
        "had_concerning_findings": len(sub_reviewer_tasks) > 0,
        "had_specialized_review": len(specialized_results) > 0,
    }

    total_extra = len(sub_results) + len(specialized_results)
    await add_transcript_entry(file_node_id, "system", f"All reviewers completed ({total_extra} additional reviewers spawned)", "message")
    await update_node_status(file_node_id, "completed", aggregated)

    if analyze_deps and all_imports and root_id:
        await spawn_dependency_analysis(node, all_imports, root_id, file_node_id, file_language, dep_depth, max_dep_depth)

    return aggregated


# ============== Dependency Analysis (from discovered imports) ==============

async def spawn_dependency_analysis(node, imports: set, root_id: str, source_file_id: str, language: str = LANGUAGE_PYTHON, depth: int = 0, max_depth: int = 2):
    """Spawn analysis for newly discovered dependencies."""
    global discovered_deps

    async with discovered_deps_lock:
        new_deps = imports - discovered_deps
        if not new_deps:
            return
        discovered_deps.update(new_deps)

    for dep_name in new_deps:
        asyncio.create_task(analyze_dependency_files(node, dep_name, root_id, source_file_id, language, depth, max_depth))


async def analyze_dependency_files(node, dep_name: str, root_id: str, discovered_in_file_id: str, language: str = LANGUAGE_PYTHON, depth: int = 0, max_depth: int = 2):
    """Analyze key files from a discovered dependency."""

    registry = "npm" if language == LANGUAGE_NODE else "PyPI"

    dep_node_id = f"dep-{dep_name}-{uuid.uuid4().hex[:6]}"
    await add_node(dep_node_id, f"dep: {dep_name}", "dependency", root_id, {"package": dep_name, "discovered_in": discovered_in_file_id, "language": language, "registry": registry, "depth": depth})
    await update_node_status(dep_node_id, "running")
    await add_transcript_entry(dep_node_id, "system", f"Fetching source files for dependency: {dep_name} (from {registry}, depth={depth})", "message")

    if language == LANGUAGE_NODE:
        files = await asyncio.to_thread(get_npm_package_files, dep_name, 3)
    else:
        files = await asyncio.to_thread(get_pypi_package_files, dep_name, 3)

    if not files or (len(files) == 1 and "error" in files[0]):
        error_msg = files[0].get("error", "Could not fetch source") if files else "No files found"
        await add_transcript_entry(dep_node_id, "system", f"Could not fetch files: {error_msg}", "message")
        await update_node_status(dep_node_id, "completed", {"package": dep_name, "error": error_msg, "language": language})
        return

    await add_transcript_entry(dep_node_id, "system", f"Found {len(files)} source files to review", "message")

    analyze_transitive = depth < max_depth

    review_tasks = []
    for file_info in files:
        task = review_file(
            node, file_info, dep_node_id,
            reviewers=["security", "quality", "complexity", "documentation"],
            analyze_deps=analyze_transitive,
            root_id=dep_node_id,
            language=language,
            dep_depth=depth + 1,
            max_dep_depth=max_depth
        )
        review_tasks.append(task)

    results = await asyncio.gather(*review_tasks, return_exceptions=True)

    dep_report = {
        "package": dep_name,
        "files_reviewed": len(files),
        "file_results": [r for r in results if isinstance(r, dict)],
        "language": language,
        "registry": registry,
        "depth": depth,
        "analyzed_transitive": analyze_transitive,
    }

    await add_transcript_entry(dep_node_id, "system", f"Completed review of {len(files)} dependency files", "message")
    await update_node_status(dep_node_id, "completed", dep_report)


# ============== Multi-Phase Analysis ==============

async def run_quick_scan_agent(node, node_id: str, file_path: str, file_content: str) -> dict:
    """Run a quick scan agent that does fast triage of a file."""

    await update_node_status(node_id, "running")
    await add_transcript_entry(node_id, "system", f"Quick scan: {file_path}", "message")

    max_content_len = 4000
    truncated = file_content[:max_content_len] if len(file_content) > max_content_len else file_content

    model = Model(
        "nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=1000,
        throttle_max_requests_in_progress=100,
        request_timeout=60.0,
    )

    agent = await Agent.start(
        node=node,
        instructions="""You are a quick code scanner. Do a fast triage of this file.
Respond with ONLY JSON:
{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "needs_full_review": true|false, "flags": ["flag1"], "summary": "one line summary"}""",
        model=model,
        max_execution_time=120.0,
        max_iterations=2,
    )

    try:
        message = f"Quick scan: {file_path}\n\n```\n{truncated}\n```"
        response = await agent.send(message, timeout=120)

        if response and len(response) > 0:
            content = response[-1].content
            result_text = content.text if hasattr(content, 'text') else str(content)
        else:
            result_text = "No response"

        await add_transcript_entry(node_id, "agent", result_text, "message")

        try:
            first_brace = result_text.find('{')
            last_brace = result_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                result = json.loads(result_text[first_brace:last_brace + 1])
            else:
                result = {"raw_response": result_text, "needs_full_review": True, "risk_level": "MEDIUM"}
        except:
            result = {"raw_response": result_text, "needs_full_review": True, "risk_level": "MEDIUM"}

        result["file"] = file_path
        result["phase"] = "quick_scan"
        await update_node_status(node_id, "completed", result)
        return result

    except Exception as e:
        await add_transcript_entry(node_id, "error", str(e), "error")
        await update_node_status(node_id, "error", {"error": str(e)})
        return {"error": str(e), "file": file_path, "needs_full_review": True, "risk_level": "MEDIUM"}
    finally:
        asyncio.create_task(Agent.stop(node, agent.name, timeout=5.0))


async def analyze_repo(node, org: str, repo: str, branch: str = None, max_files: int = 50, analyze_deps: bool = True):
    """Analyze a GitHub repository in multiple phases:

    Phase 1: Quick scan all files (1 lightweight agent each)
    Phase 2: Full review flagged files (4 reviewers each)
    Phase 3: Deep dive into high-risk files (sub-reviewers)
    """
    global discovered_deps

    await reset_graph()

    async with discovered_deps_lock:
        discovered_deps = set()

    async with graph_lock:
        graph_state["status"] = "running"

    language, detected_branch = detect_repo_language(org, repo, branch)
    if branch is None:
        branch = detected_branch
    extensions = extensions_for_language(language)

    root_id = f"root-{org}-{repo}"
    await add_node(root_id, f"{org}/{repo}", "root", None, {
        "org": org, "repo": repo, "branch": branch,
        "analyze_deps": analyze_deps, "language": language,
    })
    await update_node_status(root_id, "running")
    await add_transcript_entry(root_id, "system", f"Multi-phase analysis of {org}/{repo}@{branch}", "message")

    files = await asyncio.to_thread(get_github_files, org, repo, branch, extensions, max_files)

    if not files or (len(files) == 1 and "error" in files[0]):
        error_msg = files[0].get("error", "Unknown error") if files else "No files found"
        await add_transcript_entry(root_id, "error", f"Failed to fetch files: {error_msg}", "error")
        await update_node_status(root_id, "error", {"error": error_msg})
        async with graph_lock:
            graph_state["status"] = "completed"
        return

    await add_transcript_entry(root_id, "system", f"Found {len(files)} files for multi-phase analysis", "message")

    # ============== PHASE 1: Quick Scan ==============
    phase1_node_id = f"phase1-{uuid.uuid4().hex[:6]}"
    await add_node(phase1_node_id, "Phase 1: Quick Scan", "phase", root_id, {"phase": 1, "description": "Fast triage of all files"})
    await update_node_status(phase1_node_id, "running")
    await add_transcript_entry(phase1_node_id, "system", f"Starting quick scan of {len(files)} files", "message")

    def build_dir_tree(files):
        """Build a nested directory tree structure."""
        tree = {"children": {}, "files": []}
        for f in files:
            path_parts = f["path"].split("/")
            current = tree
            for part in path_parts[:-1]:
                if part not in current["children"]:
                    current["children"][part] = {"children": {}, "files": []}
                current = current["children"][part]
            current["files"].append(f)
        return tree

    dir_tree = build_dir_tree(files)

    quick_scan_tasks = []
    file_index_map = []
    phase1_dir_node_ids = []

    async def create_dir_nodes(tree, parent_id, path_prefix=""):
        """Recursively create directory nodes and file scanner nodes."""
        for dir_name, subtree in tree["children"].items():
            dir_path = f"{path_prefix}/{dir_name}" if path_prefix else dir_name
            dir_node_id = f"scan-dir-{uuid.uuid4().hex[:6]}"
            file_count = count_files_in_tree(subtree)
            await add_node(dir_node_id, dir_name, "directory", parent_id, {"path": dir_path, "file_count": file_count})
            await update_node_status(dir_node_id, "running")
            phase1_dir_node_ids.append(dir_node_id)
            await create_dir_nodes(subtree, dir_node_id, dir_path)

        for file_info in tree["files"]:
            scan_node_id = f"scan-{uuid.uuid4().hex[:6]}"
            short_name = os.path.basename(file_info["path"])
            await add_node(scan_node_id, short_name, "scanner", parent_id, {"path": file_info["path"]})
            task = run_quick_scan_agent(node, scan_node_id, file_info["path"], file_info["content"])
            quick_scan_tasks.append(task)
            file_index_map.append(file_info)

    def count_files_in_tree(tree):
        """Count total files in a directory tree."""
        count = len(tree["files"])
        for subtree in tree["children"].values():
            count += count_files_in_tree(subtree)
        return count

    await create_dir_nodes(dir_tree, phase1_node_id)

    scan_results = await asyncio.gather(*quick_scan_tasks, return_exceptions=True)

    for dir_node_id in phase1_dir_node_ids:
        await update_node_status(dir_node_id, "completed")

    files_needing_review = []
    high_risk_files = []
    risk_groups = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}

    for i, result in enumerate(scan_results):
        if isinstance(result, dict):
            file_info = file_index_map[i]
            risk = result.get("risk_level", "MEDIUM").upper()
            needs_review = result.get("needs_full_review", True)

            if risk not in risk_groups:
                risk = "MEDIUM"

            risk_groups[risk].append((file_info, result))

            if risk in ["HIGH", "CRITICAL"]:
                high_risk_files.append((file_info, result))
                files_needing_review.append((file_info, risk))
            elif needs_review or risk == "MEDIUM":
                files_needing_review.append((file_info, risk))

    for risk_level, risk_files in risk_groups.items():
        if risk_files:
            risk_node_id = f"risk-{risk_level.lower()}-{uuid.uuid4().hex[:6]}"
            await add_node(
                risk_node_id,
                f"{risk_level} ({len(risk_files)})",
                "risk-cluster",
                phase1_node_id,
                {"risk_level": risk_level, "file_count": len(risk_files)}
            )
            await update_node_status(risk_node_id, "completed")

    await add_transcript_entry(phase1_node_id, "system", f"Quick scan complete: {len(files_needing_review)} files need review, {len(high_risk_files)} high-risk", "message")
    await update_node_status(phase1_node_id, "completed", {
        "files_scanned": len(files),
        "files_needing_review": len(files_needing_review),
        "high_risk_files": len(high_risk_files),
        "risk_breakdown": {k: len(v) for k, v in risk_groups.items()}
    })

    # ============== PHASE 2: Full Review ==============
    if files_needing_review:
        phase2_node_id = f"phase2-{uuid.uuid4().hex[:6]}"
        await add_node(phase2_node_id, "Phase 2: Full Review", "phase", root_id, {"phase": 2, "description": "Detailed review of flagged files"})
        await add_edge(phase1_node_id, phase2_node_id)
        await update_node_status(phase2_node_id, "running")
        await add_transcript_entry(phase2_node_id, "system", f"Starting full review of {len(files_needing_review)} flagged files", "message")

        def build_review_tree(file_list):
            """Build nested directory tree for files needing review."""
            tree = {"children": {}, "files": []}
            for file_info in file_list:
                path_parts = file_info["path"].split("/")
                current = tree
                for part in path_parts[:-1]:
                    if part not in current["children"]:
                        current["children"][part] = {"children": {}, "files": []}
                    current = current["children"][part]
                current["files"].append(file_info)
            return tree

        review_tree = build_review_tree([f for f, _ in files_needing_review])

        review_tasks = []
        phase2_dir_node_ids = []

        async def create_review_dir_nodes(tree, parent_id, path_prefix=""):
            """Recursively create directory nodes for Phase 2 review."""
            for dir_name, subtree in tree["children"].items():
                dir_path = f"{path_prefix}/{dir_name}" if path_prefix else dir_name
                dir_node_id = f"dir-{uuid.uuid4().hex[:6]}"
                file_count = count_review_files(subtree)
                await add_node(dir_node_id, dir_name, "directory", parent_id, {"path": dir_path, "file_count": file_count})
                await update_node_status(dir_node_id, "running")
                phase2_dir_node_ids.append(dir_node_id)
                await create_review_dir_nodes(subtree, dir_node_id, dir_path)

            for file_info in tree["files"]:
                task = review_file(
                    node, file_info, parent_id,
                    reviewers=["security", "quality", "complexity", "documentation"],
                    analyze_deps=analyze_deps,
                    root_id=root_id,
                    language=language
                )
                review_tasks.append((parent_id, task))

        def count_review_files(tree):
            """Count files in a review tree."""
            count = len(tree["files"])
            for subtree in tree["children"].values():
                count += count_review_files(subtree)
            return count

        await create_review_dir_nodes(review_tree, phase2_node_id)

        results = await asyncio.gather(*[t[1] for t in review_tasks], return_exceptions=True)

        for dir_node_id in phase2_dir_node_ids:
            await update_node_status(dir_node_id, "completed")

        await add_transcript_entry(phase2_node_id, "system", f"Full review complete for {len(files_needing_review)} files", "message")
        await update_node_status(phase2_node_id, "completed", {"files_reviewed": len(files_needing_review)})

    # ============== PHASE 3: Deep Dive (handled by adaptive sub-reviewers) ==============
    phase3_node_id = f"phase3-{uuid.uuid4().hex[:6]}"
    await add_node(phase3_node_id, "Phase 3: Deep Dive", "phase", root_id, {"phase": 3, "description": "Sub-reviewers for high-risk findings"})
    if files_needing_review:
        await add_edge(phase2_node_id, phase3_node_id)
    else:
        await add_edge(phase1_node_id, phase3_node_id)
    await update_node_status(phase3_node_id, "running")
    await add_transcript_entry(phase3_node_id, "system", "Deep dive analysis handled by adaptive sub-reviewers during Phase 2", "message")
    await update_node_status(phase3_node_id, "completed", {"note": "Integrated with adaptive reviewers"})

    root_report = {
        "org": org,
        "repo": repo,
        "branch": branch,
        "language": language,
        "total_files": len(files),
        "phase1_scanned": len(files),
        "phase2_reviewed": len(files_needing_review) if files_needing_review else 0,
        "high_risk_count": len(high_risk_files),
    }
    await add_transcript_entry(root_id, "system", "Multi-phase analysis complete!", "message")
    await update_node_status(root_id, "completed", root_report)

    async with graph_lock:
        graph_state["status"] = "completed"


# ============== FastAPI App ==============

app = FastAPI(title="Code Review Swarm")


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.post("/analyze")
async def post_analyze(node: NodeDep, org: str = "vuejs", repo: str = "core", branch: str = None, max_files: int = 50, analyze_deps: bool = True):
    """Start multi-phase analysis of a GitHub repository."""

    async with graph_lock:
        if graph_state["status"] == "running":
            return JSONResponse(
                content={"error": "Analysis already in progress"},
                status_code=400
            )

    asyncio.create_task(analyze_repo(node, org, repo, branch, max_files, analyze_deps))

    return JSONResponse(content={"status": "started", "repo": f"{org}/{repo}", "branch": branch or "auto-detect"})


@app.get("/graph")
async def get_graph():
    """Get the current graph state."""
    async with graph_lock:
        return JSONResponse(content={
            "nodes": graph_state["nodes"],
            "edges": graph_state["edges"],
            "status": graph_state["status"],
        })


@app.get("/report/{node_id}")
async def get_report(node_id: str):
    """Get the report for a specific node."""
    async with graph_lock:
        report = graph_state["reports"].get(node_id)
        transcript = graph_state["transcripts"].get(node_id, [])
        node_info = None
        for node in graph_state["nodes"]:
            if node["id"] == node_id:
                node_info = node
                break

        if report or transcript:
            return JSONResponse(content={"node": node_info, "report": report, "transcript": transcript})
        elif node_info:
            return JSONResponse(content={"node": node_info, "report": None, "transcript": transcript, "message": "Report not yet available"})
        else:
            return JSONResponse(content={"error": "Node not found"}, status_code=404)


@app.get("/activity")
async def get_activity(limit: int = 20):
    """Get recent activity feed."""
    async with graph_lock:
        return JSONResponse(content={"activity": graph_state["activity"][:limit]})


@app.post("/reset")
async def reset():
    """Reset the graph state."""
    await reset_graph()
    return JSONResponse(content={"status": "reset"})


Node.start(http_server=HttpServer(app=app))
