import os
import tempfile
import shutil
from git import Repo
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from app.services.github_client import get_installation_token
from huggingface_hub import login
import os

login(token=os.getenv("HF_TOKEN"))


# ============================================================
# PHASE 3 — Repo Intelligence Engine
# ============================================================

class RepoIntelligenceEngine:

    def __init__(self):
        self.temp_dir = None
        self.repo_root = None
        self.file_tree = ""

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.supported_extensions = [
            ".py", ".kt", ".java", ".js", ".ts",
            ".xml", ".json", ".yml", ".yaml"
        ]

        self.chunks = []
        self.chunk_metadata = []
        self.index = None

    # --------------------------------------------------------
    # Clone Private Repo (Ephemeral)
    # --------------------------------------------------------

    def clone_repo(self, owner: str, repo: str, installation_id: int):
        self.temp_dir = tempfile.mkdtemp()

        token = get_installation_token(installation_id)

        repo_url = f"https://github.com/{owner}/{repo}.git"

        authenticated_url = repo_url.replace(
            "https://",
            f"https://x-access-token:{token}@"
        )

        clone_path = os.path.join(self.temp_dir, repo)

        Repo.clone_from(
            authenticated_url,
            clone_path,
            depth=1,
            single_branch=True
        )

        self.repo_root = clone_path

    # --------------------------------------------------------
    # Build File Tree
    # --------------------------------------------------------
    def build_file_tree(self):
        tree_lines = []

        # file extensions to ignore (images, videos, assets)
        ignore_extensions = {
            ".png", ".jpg", ".jpeg", ".webp", ".gif",
            ".mp4", ".avi", ".mov", ".mkv",
            ".mp3", ".wav",
            ".jar"
        }

        for root, dirs, files in os.walk(self.repo_root):

            # Remove hidden directories (starting with .)
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            level = root.replace(self.repo_root, "").count(os.sep)
            indent = "  " * level
            folder_name = os.path.basename(root)

            tree_lines.append(f"{indent}{folder_name}/")

            subindent = "  " * (level + 1)

            for f in files:

                # Skip hidden files
                if f.startswith("."):
                    continue

                # Skip images, videos, binary assets
                if any(f.lower().endswith(ext) for ext in ignore_extensions):
                    continue

                tree_lines.append(f"{subindent}{f}")

        self.file_tree = "\n".join(tree_lines)
    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    def chunk_file(self, file_path, chunk_size=2000, overlap=200):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    # --------------------------------------------------------
    # Process Repository
    # --------------------------------------------------------

    def process_repository(self):
        self.chunks = []
        self.chunk_metadata = []

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d != ".git"]

            for file in files:
                if any(file.endswith(ext) for ext in self.supported_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.repo_root)

                    file_chunks = self.chunk_file(full_path)

                    for chunk in file_chunks:
                        self.chunks.append(chunk)
                        self.chunk_metadata.append({
                            "file": relative_path
                        })

        if not self.chunks:
            return False

        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.build_file_tree()

        return True

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    def search(self, query, top_k=5):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx in indices[0]:
            results.append({
                "file": self.chunk_metadata[idx]["file"],
                "code": self.chunks[idx]
            })

        return results

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)