import json
import os
import re
from google import genai


class RootCauseEngine:

    def __init__(self, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name

    def _format_context(self, repo_context):
        formatted = ""
        for i, chunk in enumerate(repo_context):
            formatted += f"\n--- FILE {i+1}: {chunk['file']} ---\n"
            formatted += chunk["code"]
            formatted += "\n"
        return formatted

    def _safe_json_parse(self, text):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            raise

    def analyze(self, issue_title, issue_body, repo_context, file_tree):

        if not repo_context:
            return {
                "root_cause_summary": "No relevant code retrieved.",
                "likely_files": [],
                "reasoning": "Phase 3 retrieval returned empty.",
                "fix_strategy": "Increase retrieval depth.",
                "confidence": 0.0
            }

        formatted_context = self._format_context(repo_context)

        prompt = f"""
You are a senior software maintenance engineer.

Rules:
- Only use provided repository context.
- Do NOT invent files.
- Only reference files that appear in REPOSITORY CONTEXT.
- If insufficient data, say so.
- Provide the suggested code changes in fix_strategy.
- Return strict JSON only.
- Keep it concise.

ISSUE TITLE:
{issue_title}

ISSUE DESCRIPTION:
{issue_body}

REPOSITORY CONTEXT:
{formatted_context}

Return JSON:

{{
  "root_cause_summary": "...",
  "likely_files": ["..."],
  "reasoning": "...",
  "fix_strategy": "...",
  "confidence": 0.0
}}
"""
        token_info = self.client.models.count_tokens(
            model=self.model_name,
            contents=prompt
        )

        input_tokens = token_info.total_tokens

        print(f"[LLM] Input tokens: {input_tokens}")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        try:
            return self._safe_json_parse(response.text)
        except json.JSONDecodeError:
            return {
                "root_cause_summary": "Invalid JSON from model.",
                "likely_files": [],
                "input_tokens": input_tokens,
                "prompt":prompt,
                "reasoning": response.text,
                "fix_strategy": "Prompt tuning required.",
                "confidence": 0.0
            }
        except Exception as e:
            return {
                "root_cause_summary": "LLM failure.",
                "prompt":prompt,
                "likely_files": [],
                "input_tokens": input_tokens,
                "reasoning": str(e),
                "fix_strategy": "Model or API failure.",
                "confidence": 0.0
            }