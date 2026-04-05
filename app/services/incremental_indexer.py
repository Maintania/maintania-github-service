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
    
    
    def _get_extensions_from_changes(self, changed_files):
        exts = set()

        for f in changed_files:
            path = f["path"]
            _, ext = os.path.splitext(path)
            if ext:
                exts.add(ext.lower())

        return list(exts)
    # -----------------------------
    # GIT DIFF
    # -----------------------------
    def get_changed_files(self, last_commit):
        print("[IncrementalIndexer] Step: calculating changed files")

        repo = Repo(self.engine.repo_root)
        current_commit = repo.head.commit.hexsha
        print(f"[IncrementalIndexer] Current commit: {current_commit}")

        if not last_commit:
            print("[IncrementalIndexer] No last commit found, full reindex needed")
            return "FULL", current_commit

        if last_commit == current_commit:
            print("[IncrementalIndexer] Last commit matches current commit, no changes")
            return [], current_commit

        # -----------------------------------
        # FIX: ensure last_commit exists locally
        # -----------------------------------
        try:
            repo.git.cat_file("-e", last_commit)
        except Exception:
            print("[IncrementalIndexer] Last commit missing in shallow clone, fetching more history")

            # fetch deeper history
            repo.git.fetch("--depth=1000")

            try:
                repo.git.cat_file("-e", last_commit)
            except Exception:
                print("[IncrementalIndexer] Commit still missing after fetch, fallback to full reindex")
                return "FULL", current_commit

        # -----------------------------------
        # SAFE DIFF
        # -----------------------------------
        diff_output = repo.git.diff(
            f"{last_commit}..{current_commit}",
            name_status=True
        ).splitlines()
        print(f"[IncrementalIndexer] Diff entries found: {len(diff_output)}")

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
    def process_changed_files(self, owner, repo, branch, changed_files, allowed_exts):
        print(f"[IncrementalIndexer] Step: processing {len(changed_files)} changed files")

        buffer_chunks = []
        buffer_meta = []
        total_files = len(changed_files)

        # -----------------------------------
        # PREPARE DELETE PATHS (ONLY VALID FILES)
        # -----------------------------------
        delete_paths = []

        for file_info in changed_files:
            relative_path = file_info["path"]
            status = file_info["status"]

            _, ext = os.path.splitext(relative_path)
            ext = ext.lower()

            if ext and ext not in allowed_exts:
                continue

            if self.engine.should_ignore_file(os.path.basename(relative_path)):
                continue

            if status != "D":  # only delete existing/modified files
                delete_paths.append(relative_path)

        # -----------------------------------
        # BULK DELETE (ONLY ONCE)
        # -----------------------------------
        if delete_paths:
            print(f"[IncrementalIndexer] Bulk deleting {len(delete_paths)} files")
            self.engine.delete_files_parallel(owner, repo, branch, delete_paths)

        # -----------------------------------
        # PROCESS FILES
        # -----------------------------------
        for idx, file_info in enumerate(changed_files, 1):

            relative_path = file_info["path"]
            status = file_info["status"]

            _, ext = os.path.splitext(relative_path)
            ext = ext.lower()

            # FILTER
            if ext and ext not in allowed_exts:
                print(f"[IncrementalIndexer] [{idx}/{total_files}] Skipping (filtered ext): {relative_path}")
                continue

            if self.engine.should_ignore_file(os.path.basename(relative_path)):
                print(f"[IncrementalIndexer] [{idx}/{total_files}] Skipping (ignored file): {relative_path}")
                continue

            print(f"[IncrementalIndexer] [{idx}/{total_files}] File: {relative_path} (status={status})")

            # HANDLE DELETED FILE
            if status == "D":
                print(f"[IncrementalIndexer] Skipping deleted file: {relative_path}")
                continue

            full_path = os.path.join(self.engine.repo_root, relative_path)

            if not os.path.exists(full_path):
                print(f"[IncrementalIndexer] Skipping missing file on disk: {relative_path}")
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

            # BATCH FLUSH
            if len(buffer_chunks) >= self.engine.embed_batch_size:
                print(f"[IncrementalIndexer] [{idx}/{total_files}] Flushing batch ({len(buffer_chunks)} chunks)")
                self.engine.safe_process_batch(buffer_chunks, buffer_meta)
                buffer_chunks, buffer_meta = [], []

        # FINAL FLUSH
        if buffer_chunks:
            print(f"[IncrementalIndexer] Final flush with {len(buffer_chunks)} chunks")
            self.engine.safe_process_batch(buffer_chunks, buffer_meta)

    # -----------------------------
    # MAIN ENTRYPOINT
    # -----------------------------
    def run(self, owner, repo, branch):
        print(f"[IncrementalIndexer] Start incremental indexing for {owner}/{repo} (branch={branch})")

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
            print("[IncrementalIndexer] Step: reading last indexed commit")
            last_commit = self.get_last_commit(owner, repo, branch)
            print(f"[IncrementalIndexer] Last commit from state: {last_commit}")

            print("[IncrementalIndexer] Step: computing repository diff")
            changed_files, current_commit = self.get_changed_files(last_commit)

            # First time indexing
            if changed_files == "FULL":
                print("[IncrementalIndexer] Full reindex required")
                return "FULL_REINDEX"

            if not changed_files:
                print("[IncrementalIndexer] No changes detected, marking state as ready")
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
                print("[IncrementalIndexer] Finished: NO_CHANGE")
                return "NO_CHANGE"

            print(f"[IncrementalIndexer] Changed files count: {len(changed_files)}")

            print("[IncrementalIndexer] Step: filtering extensions")

            extensions = self._get_extensions_from_changes(changed_files)

            print(f"[IncrementalIndexer] Extensions found: {extensions}")

            llm_allowed_exts = self.engine.ask_llm_for_extensions(extensions)

            print(f"[IncrementalIndexer] Allowed extensions: {llm_allowed_exts}")

            print("[IncrementalIndexer] Step: processing changed files")
            self.process_changed_files(owner, repo, branch, changed_files, llm_allowed_exts)
            
            print("[IncrementalIndexer] Step: updating last commit")
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

            print("[IncrementalIndexer] Finished: UPDATED")
            return "UPDATED"

        except Exception as e:
            print(f"[IncrementalIndexer] Failed with error: {e}")
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
