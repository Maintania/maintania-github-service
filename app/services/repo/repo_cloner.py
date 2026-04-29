import os
import tempfile
import shutil
import uuid
import json
import time
import re
from datetime import datetime, timezone
import requests
import yaml
import numpy as np
from git import Repo
from sentence_transformers import SentenceTransformer
from tree_sitter_languages import get_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from app.services.github.github_client import get_installation_token
import torch
from google import genai
from app.services.ai.llm_client import LLMClient

# ===============================
# GPU TOGGLE
# ===============================
USE_GPU = True  # set False to force CPU
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
client = genai.Client(api_key=GEMINI_API_KEY)

llm = LLMClient()

if USE_GPU and torch.cuda.is_available():
    DEVICE = "cuda"
    print("Embedding model running on GPU")
else:
    DEVICE = "cpu"
    print("Embedding model running on CPU")


MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-2.5-flash-lite"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

MAX_WORKERS = 4
# CHANGED PARAMETERS
EMBED_BATCH_SIZE = 512
MODEL_BATCH_SIZE = 256

MODEL_BATCH_SIZE_CPU = 64
MODEL_BATCH_SIZE_GPU = 128

MAX_FILE_SIZE = 200_000

LINGUIST_URL = "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml"
LINGUIST_CACHE = "linguist_extensions.json"

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

def embed(texts):

    batch_size = MODEL_BATCH_SIZE_GPU if DEVICE == "cuda" else MODEL_BATCH_SIZE_CPU

    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size
    )

def clean_llm_json(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return json.loads(match.group(0))
        return []
    
    
def load_extension_map():

    if os.path.exists(LINGUIST_CACHE):
        with open(LINGUIST_CACHE, "r") as f:
            return json.load(f)

    response = requests.get(LINGUIST_URL)
    response.raise_for_status()

    languages = yaml.safe_load(response.text)

    extension_map = {}

    for lang_name, data in languages.items():
        for ext in data.get("extensions", []):
            extension_map[ext] = lang_name.lower().replace(" ", "_")

    with open(LINGUIST_CACHE, "w") as f:
        json.dump(extension_map, f)

    return extension_map


extension_map = load_extension_map()


HIGH_VALUE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".cpp", ".c", ".h",
    ".cs", ".rs", ".php", ".rb", ".kt"
}

LOW_VALUE_EXTENSIONS = {
    ".po", ".csv", ".log", ".txt", ".lock", ".map"
}

CONDITIONAL_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".md"
}


class RepoIntelligenceEngine:

    def __init__(self):

        self.temp_dir = None
        self.repo_root = None

        self.parser_cache = {}

        self.ignore_dirs = {
            ".git",
            ".idea",
            "build",
            ".gradle",
            "node_modules",
            "dist",
            "__pycache__",
            "test",
            "tests",
            "androidTest",
            "docs",
            "doc",
            "static",
            "migrations",
            "examples",
            "demo",
            "samples",
            "benchmark",
            "benchmarks"
        }

        self.ignore_file_names = {
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "Pipfile.lock",
            "composer.lock",
            "Cargo.lock",
            ".DS_Store",
            "Thumbs.db",
            "__init__.py"
        }

        self.ignore_extensions = {
            ".png",".jpg",".jpeg",".gif",".bmp",".ico",".pdf",
            ".zip",".tar",".gz",".rar",".7z",
            ".exe",".dll",".so",".dylib",".class",".o",".a",".lib",
            ".db",".sqlite",".sqlite3",".parquet",".csv",
            ".log",".lock",".map",
            ".min.js",".bundle.js"
        }

        self.qdrant = QdrantClient(
            url=os.getenv("Qdrant_URL"), 
            api_key=os.getenv("Qdrant_Api_Key"),
        )

        self.collection_name = "repo_code_embeddings"
        self.state_collection = "repo_index_state"

        self.embed_batch_size = EMBED_BATCH_SIZE
        
    def get_uncertain_extensions(self, repo_map):
        exts = set()

        for f in repo_map["files"]:
            _, ext = os.path.splitext(f["path"])
            ext = ext.lower()

            if (
                ext
                and ext not in HIGH_VALUE_EXTENSIONS
                and ext not in LOW_VALUE_EXTENSIONS
            ):
                exts.add(ext)

        return list(exts)
    
    def llm_filter_extensions(self,extensions: list[str]):

        if not GEMINI_API_KEY:
            return [],0
        print(f'Extension filtering with {len(extensions)} extensions: {", ".join(extensions)}')
        prompt = f"""
        Given these file extensions from a GitHub repository:

        {extensions}

        Select ONLY the extensions useful for:
        - understanding code logic
        - debugging
        - fixing issues

        Ignore:
        - translations (.po)
        - logs
        - datasets
        - generated files

        Only necessary extensions should be selected.

        Strictly return JSON list:
        [".ext1", ".ext2"]
        """
        token_count = 0

        # ---------------------------
        # Token Count
        # ---------------------------
        try:
            token_data = llm.count_tokens("gemini", GEMINI_MODEL, prompt)
            token_count = token_data["total_tokens"]
            print("Gemini extension filter tokens:", token_count)
        except Exception as e:
            print("Token count failed:", e)


        # ---------------------------
        # LLM Call (with retry)
        # ---------------------------
        try:
            result_obj = None

            for _ in range(2):
                try:
                    result_obj = llm.generate(
                        "gemini",
                        GEMINI_MODEL,
                        prompt,
                        config={
                            "temperature": 0,
                            "max_output_tokens": 100
                        }
                    )
                    break
                except Exception:
                    time.sleep(1)

            if not result_obj:
                return [], token_count

            text = result_obj["text"]
            print("Raw Gemini response:", text)

            llm_output = clean_llm_json(text)

            if not isinstance(llm_output, list):
                return [], token_count

            valid_input = set(extensions)

            filtered = [
                e.lower()
                for e in llm_output
                if isinstance(e, str) and e.lower() in valid_input
            ]

            return filtered, token_count

        except Exception as e:
            print("LLM extension filter failed:", e)
            return [], token_count
        
    
    def ask_llm_for_extensions(self, extensions):
        
        if not extensions:
            return set()

        extensions = extensions[:50]

        try:
            llm_selected, _ = self.llm_filter_extensions(extensions)
            return set(llm_selected)
        except Exception as e:
            print("LLM failed:", e)
            return set()


    def build_extension_filter(self, repo_map):
        uncertain = self.get_uncertain_extensions(repo_map)

        llm_selected = self.ask_llm_for_extensions(uncertain)

        final_exts = set(HIGH_VALUE_EXTENSIONS)
        final_exts.update(llm_selected)

        return final_exts

    def _utc_now_iso(self):
        return datetime.now(timezone.utc).isoformat()


    def _generate_repo_id(self, owner, repo, branch):
        key = f"{owner}/{repo}:{branch}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _process_single_file(self, file_info, owner, repo, branch):

        relative_path = file_info["path"]
        language = file_info["language"]

        full_path = os.path.join(self.repo_root, relative_path)

        if not os.path.exists(full_path):
            return [], []

        buffer_chunks = []
        buffer_meta = []

        symbols = self.extract_symbols(full_path, language)

        # ---- SYMBOL PATH ----
        if symbols:
            for symbol in symbols:

                chunks = self.chunk_symbol(
                    symbol["code"],
                    symbol["start_line"]
                )

                for c in chunks:
                    buffer_chunks.append(c["text"])
                    buffer_meta.append({
                        "repo": f"{owner}/{repo}",
                        "branch": branch,
                        "file": relative_path,
                        "language": language,
                        "symbol_type": symbol["type"],
                        "symbol_name": symbol["name"],
                        "start_line": c["line"]
                    })

        # ---- FILE CHUNK PATH ----
        else:
            file_chunks = self.chunk_file(full_path)

            for chunk in file_chunks:
                buffer_chunks.append(chunk)
                buffer_meta.append({
                    "repo": f"{owner}/{repo}",
                    "branch": branch,
                    "file": relative_path,
                    "language": language,
                    "symbol_type": "file_chunk",
                    "symbol_name": None,
                    "start_line": 0
                })

        return buffer_chunks, buffer_meta
    
    
    def get_parser_for_language(self, language):

        if language in self.parser_cache:
            return self.parser_cache[language]

        try:
            parser = get_parser(language)
            self.parser_cache[language] = parser
            return parser
        except Exception:
            return None

    def detect_language(self, file_name):

        _, ext = os.path.splitext(file_name)

        return extension_map.get(ext.lower(), "unknown")

    def should_ignore_file(self, file_name):

        if file_name in self.ignore_file_names:
            return True

        for ext in self.ignore_extensions:
            if file_name.endswith(ext):
                return True

        return False

    def extract_symbols(self, file_path, language):

        try:
            if os.path.getsize(file_path) < 300:
                return []
        except:
            return []

        parser = self.get_parser_for_language(language)

        if not parser:
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except:
            return []

        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        symbols = []
        stack = [root]

        SYMBOL_TYPES = {
            "function_definition",
            "method_definition",
            "class_definition",
            "interface_declaration",
            "struct_specifier",
            "function_declaration"
        }

        while stack:

            node = stack.pop()

            if node.type in SYMBOL_TYPES:

                start_byte = node.start_byte
                end_byte = node.end_byte

                if end_byte <= start_byte:
                    stack.extend(node.children)
                    continue

                symbol_code = code[start_byte:end_byte].strip()

                if len(symbol_code) < 80:
                    stack.extend(node.children)
                    continue

                if symbol_code.startswith("}") or symbol_code.startswith(")"):
                    stack.extend(node.children)
                    continue

                name_node = node.child_by_field_name("name")

                if name_node:
                    name = code[name_node.start_byte:name_node.end_byte]
                else:
                    name = "unknown"

                symbols.append({
                    "type": node.type,
                    "name": name,
                    "start_line": node.start_point[0],
                    "code": symbol_code
                })

            stack.extend(node.children)

        return symbols

    def chunk_symbol(self, symbol_code, base_line):

        lines = symbol_code.split("\n")

        chunks = []

        window = 40
        overlap = 10

        start = 0

        while start < len(lines):

            end = start + window

            chunk_lines = lines[start:end]

            chunk_text = "\n".join(chunk_lines).strip()

            if len(chunk_text) > 20:
                chunks.append({
                    "text": chunk_text,
                    "line": base_line + start
                })

            start += window - overlap

        return chunks

    def chunk_file(self, file_path):

        try:

            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return []

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        except Exception:
            return []

        chunks = []

        start = 0

        while start < len(content):

            end = start + CHUNK_SIZE
            chunk = content[start:end]

            chunks.append(chunk)

            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    def build_repo_structure_map(self, owner, repo, branch):
        repo_map = {
            "repo": f"{owner}/{repo}",
            "branch": branch,
            "last_commit": None,
            "files": []
        }

        # -----------------------------
        # STEP 1: COLLECT FILES
        # -----------------------------
        for root, dirs, files in os.walk(self.repo_root):

            # remove ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for file in files:

                if self.should_ignore_file(file):
                    continue

                full_path = os.path.join(root, file)

                try:
                    size = os.path.getsize(full_path)
                except:
                    continue

                if size > MAX_FILE_SIZE:
                    continue

                relative_path = os.path.relpath(full_path, self.repo_root)
                language = self.detect_language(file)

                repo_map["files"].append({
                    "path": relative_path,
                    "language": language,
                    "size": size
                })

        print("FILES BEFORE FILTER:", len(repo_map["files"]))

        # -----------------------------
        # STEP 2: EXTENSION FILTER
        # -----------------------------
        allowed_extensions = self.build_extension_filter(repo_map)

        filtered_files = []

        for f in repo_map["files"]:
            _, ext = os.path.splitext(f["path"])
            ext = ext.lower()

            # ❌ skip low-value
            if ext in LOW_VALUE_EXTENSIONS:
                continue

            # ✅ always include high-value
            if ext in HIGH_VALUE_EXTENSIONS:
                filtered_files.append(f)
                continue

            # ⚖️ conditional include
            if ext in CONDITIONAL_EXTENSIONS:
                if f["size"] < MAX_FILE_SIZE:
                    filtered_files.append(f)
                continue

            # 🤖 LLM-selected extensions
            if ext in allowed_extensions:
                filtered_files.append(f)

        repo_map["files"] = filtered_files

        print("FILES AFTER FILTER:", len(repo_map["files"]))

        return repo_map
 
 
    def reset_collection(self):
        collections = self.qdrant.get_collections().collections
        if self.collection_name in [c.name for c in collections] or self.state_collection in [c.name for c in collections]:
            self.qdrant.delete_collection(collection_name=self.collection_name)
            self.qdrant.delete_collection(collection_name=self.state_collection)

            
    def create_state_collection(self):

        collections = self.qdrant.get_collections().collections

        if self.state_collection not in [c.name for c in collections]:

            self.qdrant.create_collection(
                collection_name=self.state_collection,
                vectors_config=VectorParams(
                    size=1,
                    distance=Distance.COSINE
                )
            )

    def upsert_repo_state(self, owner, repo, branch, data):

        self.create_state_collection()

        repo_id = self._generate_repo_id(owner, repo, branch)
        existing = self.get_repo_state(owner, repo, branch) or {}

        payload = {
            **existing,
            **data
        }

        payload["repo"] = f"{owner}/{repo}"
        payload["branch"] = branch
        
        self.qdrant.upsert(
            collection_name=self.state_collection,
            points=[
                PointStruct(
                    id=repo_id,
                    vector=[0.0],
                    payload=payload
                )
            ]
        )

    def get_repo_state(self, owner, repo, branch):

        self.create_state_collection()

        repo_id = self._generate_repo_id(owner, repo, branch)

        result = self.qdrant.retrieve(
            collection_name=self.state_collection,
            ids=[repo_id]
        )

        if not result:
            return None

        return result[0].payload

    def get_last_commit(self, owner, repo, branch):

        state = self.get_repo_state(owner, repo, branch)
        if not state:
            return None
        return state.get("last_commit")

    def update_last_commit(self, owner, repo, branch, commit):

        self.upsert_repo_state(
            owner,
            repo,
            branch,
            {
                "last_commit": commit
            }
        )

    def delete_repo_state(self, owner, repo, branch):

        repo_id = self._generate_repo_id(owner, repo, branch)

        self.qdrant.delete(
            collection_name=self.state_collection,
            points_selector=[repo_id]
        )

    def delete_state(self, owner, repo, branch=None):

        if branch:
            self.delete_repo_state(owner, repo, branch)
        else:
            self.qdrant.delete(
                collection_name=self.state_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="repo",
                            match=MatchValue(value=f"{owner}/{repo}")
                        )
                    ]
                )
            )


    def delete_repo(self, owner: str, repo: str, branch: str | None = None):
        repo_name = f"{owner}/{repo}"
        conditions = [
            FieldCondition(
                key="repo",
                match=MatchValue(value=repo_name)
            )
        ]
        if branch:
            conditions.append(
                FieldCondition(
                    key="branch",
                    match=MatchValue(value=branch)
                )
            )
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions)
        )
    def create_payload_indexes(self):
        print("[Qdrant] Creating payload indexes...")

        self.qdrant.create_payload_index(
            collection_name=self.collection_name,
            field_name="repo",
            field_schema="keyword"
        )

        self.qdrant.create_payload_index(
            collection_name=self.collection_name,
            field_name="branch",
            field_schema="keyword"
        )

        self.qdrant.create_payload_index(
            collection_name=self.collection_name,
            field_name="file",
            field_schema="keyword"
        )

        print("[Qdrant] Payload indexes created")
    def create_collection(self, vector_size):

        collections = self.qdrant.get_collections().collections

        if self.collection_name not in [c.name for c in collections]:

            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            self.create_payload_indexes()

    def process_repository(self, owner, repo, branch):

        start_time = time.time()

        vector_size = model.get_sentence_embedding_dimension()
        self.create_collection(vector_size)

        repo_map = self.build_repo_structure_map(owner, repo, branch)

        buffer_chunks = []
        buffer_meta = []

        total_chunks = 0
        total_files = 0
        language_counts = {}

        counter = 0
        last_report_time = start_time

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            futures = [
                executor.submit(
                    self._process_single_file,
                    file_info,
                    owner,
                    repo,
                    branch
                )
                for file_info in repo_map["files"]
            ]

            for i, future in enumerate(as_completed(futures), 1):

                chunks, metas = future.result()
                lang = metas[0]["language"] if metas else "unknown"
                language_counts[lang] = language_counts.get(lang, 0) + 1
                buffer_chunks.extend(chunks)
                buffer_meta.extend(metas)

                total_chunks += len(chunks)
                total_files += 1

                # batching stays SAME
                if len(buffer_chunks) >= EMBED_BATCH_SIZE:
                    self.safe_process_batch(buffer_chunks, buffer_meta)
                    buffer_chunks = []
                    buffer_meta = []

                # logging stays similar
                if i % 100 == 0:
                    now = time.time()
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"files={i} chunks={total_chunks} "
                        f"time={(now-last_report_time):.2f}s"
                    )
                    last_report_time = now

        if buffer_chunks:
            self.safe_process_batch(buffer_chunks, buffer_meta)

        print("FILES INDEXED:", total_files)
        print("CHUNKS INDEXED:", total_chunks)
        print("TOTAL TIME:", time.time() - start_time)

        index_duration = round(time.time() - start_time, 2)

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "languages": language_counts,
            "duration_sec": index_duration
        }, True
    def _process_batch(self, chunks, metadata):

        embeddings = embed(chunks)
        embeddings = np.array(embeddings).astype("float32")

        batch_size = 100  # 🔥 tune this (100–500 ideal)

        for start in range(0, len(embeddings), batch_size):

            end = start + batch_size

            batch_vectors = embeddings[start:end]
            batch_chunks = chunks[start:end]
            batch_meta = metadata[start:end]

            points = []

            for i, vector in enumerate(batch_vectors):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.tolist(),
                        payload={
                            "repo": batch_meta[i]["repo"],
                            "branch": batch_meta[i]["branch"],
                            "file": batch_meta[i]["file"],
                            "language": batch_meta[i]["language"],
                            "symbol_type": batch_meta[i]["symbol_type"],
                            "start_line": batch_meta[i]["start_line"],
                            "code": batch_chunks[i]
                        }
                    )
                )

            # 🚀 batched upsert
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
    def safe_process_batch(self, chunks, meta):
        try:
            self._process_batch(chunks, meta)

        except Exception as e:
            print(f"[Batch Failed] size={len(chunks)} error={e}")

            if len(chunks) <= 32:
                print("[Batch Failed] Skipping small batch")
                return

            mid = len(chunks) // 2

            self.safe_process_batch(chunks[:mid], meta[:mid])
            self.safe_process_batch(chunks[mid:], meta[mid:])
        
    def resolve_branch(self, owner, repo, installation_id, branch=None):

        token = get_installation_token(installation_id)

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        repo_api = f"https://api.github.com/repos/{owner}/{repo}"

        response = requests.get(repo_api, headers=headers)
        response.raise_for_status()

        default_branch = response.json()["default_branch"]

        return branch if branch else default_branch
    
    
    def clone_repo(self, owner: str, repo: str, installation_id: int, branch: str | None = None):

        self.temp_dir = tempfile.mkdtemp()

        token = get_installation_token(installation_id)

        if not token or len(token) < 20:
            raise Exception("Invalid GitHub installation token")

        repo_api = f"https://api.github.com/repos/{owner}/{repo}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        # -----------------------------
        # Validate token
        # -----------------------------
        # test_resp = requests.get("https://api.github.com/user", headers=headers)
        # if test_resp.status_code != 200:
        #     raise Exception(f"GitHub token invalid: {test_resp.status_code} {test_resp.text}")
        repo_api = f"https://api.github.com/repos/{owner}/{repo}"

        response = requests.get(repo_api, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Repo access failed: {response.status_code} {response.text}")
        # -----------------------------
        # Validate repo access
        # -----------------------------
        response = requests.get(repo_api, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Repo access failed: {response.status_code} {response.text}")

        repo_data = response.json()
        default_branch = repo_data["default_branch"]

        # Use provided branch if exists, else fallback
        target_branch = branch if branch else default_branch
        
        repo_url = f"https://github.com/{owner}/{repo}.git"

        authenticated_url = repo_url.replace(
            "https://",
            f"https://x-access-token:{token}@"
        )

        clone_path = os.path.join(self.temp_dir, repo)

        # -----------------------------
        # RETRY LOGIC
        # -----------------------------
        last_error = None

        for attempt in range(3):
            try:
                print(f"Clone attempt {attempt+1}...")

                Repo.clone_from(
                    authenticated_url,
                    clone_path,
                    depth=1,
                    single_branch=True,
                    branch=target_branch,
                    env={"GIT_CLONE_PROTECTION_ACTIVE": "false"}
                )

                self.repo_root = clone_path
                return target_branch

            except Exception as e:
                print(f"Auth clone failed: {e}")
                last_error = e
                time.sleep(2)

        # -----------------------------
        # FALLBACK (public repos)
        # -----------------------------
        try:
            print("Falling back to public clone...")

            Repo.clone_from(
                repo_url,
                clone_path,
                depth=1,
                single_branch=True,
                branch=target_branch
            )

            self.repo_root = clone_path
            return target_branch

        except Exception as e:
            raise Exception(
                f"Clone failed after retries + fallback:\nAuth error: {last_error}\nPublic error: {e}"
            )
    
    from concurrent.futures import ThreadPoolExecutor

    def delete_files_parallel(self, owner, repo, branch, file_paths):
        def delete_one(path):
            try:
                self.delete_file_vectors(owner, repo, branch, path)
            except Exception as e:
                print(f"[Delete Error] {path}: {e}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(delete_one, file_paths)
    
    def delete_file_vectors(self, owner, repo, branch, file_path):
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="repo",
                        match=MatchValue(value=f"{owner}/{repo}")
                    ),
                    FieldCondition(
                        key="branch",
                        match=MatchValue(value=branch)
                    ),
                    FieldCondition(
                        key="file",
                        match=MatchValue(value=file_path)
                    )
                ]
            )
        )

    def cleanup(self):

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
