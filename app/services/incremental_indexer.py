import os
import json
import time
from git import Repo
from app.services.repo_cloner import *

STATE_FILE = "repo_index_state.json"


class IncrementalIndexer:

    def __init__(self, engine):
        self.engine = engine
        self.state = self._load_state()

    # -----------------------------
    # STATE MANAGEMENT
    # -----------------------------
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _get_key(self, owner, repo, branch):
        return f"{owner}/{repo}:{branch}"

    def get_last_commit(self, owner, repo, branch):
        return self.engine.get_last_commit(owner, repo, branch)

    def update_commit(self, owner, repo, branch, commit):
        self.engine.update_last_commit(owner, repo, branch, commit)

    # -----------------------------
    # GIT DIFF
    # -----------------------------
    def get_changed_files(self, last_commit):

        repo = Repo(self.engine.repo_root)
        current_commit = repo.head.commit.hexsha

        if not last_commit:
            return "FULL", current_commit

        if last_commit == current_commit:
            return [], current_commit

        # -----------------------------------
        # FIX: ensure last_commit exists locally
        # -----------------------------------
        try:
            repo.git.cat_file("-e", last_commit)
        except Exception:
            print("Last commit not in shallow clone → fetching more history...")

            # fetch deeper history
            repo.git.fetch("--depth=1000")

            try:
                repo.git.cat_file("-e", last_commit)
            except Exception:
                print("Still missing commit → fallback to FULL reindex")
                return "FULL", current_commit

        # -----------------------------------
        # SAFE DIFF
        # -----------------------------------
        diff_output = repo.git.diff(
            f"{last_commit}..{current_commit}",
            name_status=True
        ).splitlines()

        changed_files = []

        for line in diff_output:
            parts = line.split("\t")
            status = parts[0]
            file_path = parts[1]

            changed_files.append({
                "status": status,
                "path": file_path
            })

        return changed_files, current_commit

    # -----------------------------
    # PROCESS CHANGES
    # -----------------------------
    def process_changed_files(self, owner, repo, branch, changed_files):

        buffer_chunks = []
        buffer_meta = []

        for file_info in changed_files:

            relative_path = file_info["path"]
            status = file_info["status"]

            full_path = os.path.join(self.engine.repo_root, relative_path)

            # -----------------------------------
            # ALWAYS delete old vectors
            # -----------------------------------
            self.engine.delete_file_vectors(owner, repo, branch, relative_path)

            # -----------------------------------
            # HANDLE DELETED FILE
            # -----------------------------------
            if status == "D":
                continue

            if not os.path.exists(full_path):
                continue

            language = self.engine.detect_language(relative_path)

            symbols = self.engine.extract_symbols(full_path, language)

            if symbols:
                for symbol in symbols:

                    chunks = self.engine.chunk_symbol(
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

            else:
                chunks = self.engine.chunk_file(full_path)

                for chunk in chunks:
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

            # -----------------------------------
            # BATCH FLUSH
            # -----------------------------------
            if len(buffer_chunks) >= self.engine.embed_batch_size:
                self.engine._process_batch(buffer_chunks, buffer_meta)
                buffer_chunks, buffer_meta = [], []

        if buffer_chunks:
            self.engine._process_batch(buffer_chunks, buffer_meta)

    # -----------------------------
    # MAIN ENTRYPOINT
    # -----------------------------
    def run(self, owner, repo, branch):

        start_time = time.time()
        self.engine.upsert_repo_state(
            owner,
            repo,
            branch,
            {
                "status": "updating",
                "last_update_type": "incremental",
                "error": None
            }
        )

        try:
            last_commit = self.get_last_commit(owner, repo, branch)
            changed_files, current_commit = self.get_changed_files(last_commit)

            # First time indexing
            if changed_files == "FULL":
                print("No previous state -> Full indexing required")
                return "FULL_REINDEX"

            if not changed_files:
                print("No changes detected")
                self.engine.upsert_repo_state(
                    owner,
                    repo,
                    branch,
                    {
                        "last_commit": current_commit,
                        "last_indexed_at": self.engine._utc_now_iso(),
                        "last_index_duration_sec": round(time.time() - start_time, 2),
                        "last_update_type": "incremental",
                        "status": "ready",
                        "error": None
                    }
                )
                return "NO_CHANGE"

            print(f"Changed files: {len(changed_files)}")

            self.process_changed_files(owner, repo, branch, changed_files)
            self.update_commit(owner, repo, branch, current_commit)

            self.engine.upsert_repo_state(
                owner,
                repo,
                branch,
                {
                    "last_commit": current_commit,
                    "last_indexed_at": self.engine._utc_now_iso(),
                    "last_index_duration_sec": round(time.time() - start_time, 2),
                    "last_update_type": "incremental",
                    "status": "ready",
                    "error": None
                }
            )

            return "UPDATED"

        except Exception as e:
            self.engine.upsert_repo_state(
                owner,
                repo,
                branch,
                {
                    "status": "failed",
                    "last_indexed_at": self.engine._utc_now_iso(),
                    "last_index_duration_sec": round(time.time() - start_time, 2),
                    "last_update_type": "incremental",
                    "error": str(e)
                }
            )
            raise
