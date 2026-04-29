import json
import os
import re
from google import genai
from app.services.ai.llm_client import LLMClient


class RootCauseEngine:

    def __init__(self, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.llm = LLMClient()

    # ---------------------------
    # Context Formatting (Improved)
    # ---------------------------
    def _format_context(self, repo_context, max_chunks=6):
        # sort by score if available
        repo_context = sorted(
            repo_context,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:max_chunks]

        formatted = ""

        for i, chunk in enumerate(repo_context):
            formatted += f"\n--- FILE {i+1}: {chunk['file']} ---\n"
            formatted += f"Lines: {chunk.get('start_line', 0)}\n"
            formatted += f"Type: {chunk.get('symbol_type', 'unknown')}\n"
            formatted += f"Symbol: {chunk.get('symbol_name', 'unknown')}\n\n"

            # truncate code to avoid token explosion
            code = chunk["code"][:800]
            formatted += code + "\n"

        return formatted

    # ---------------------------
    # Safer JSON Parsing
    # ---------------------------
    def _safe_json_parse(self, text):
        if not text:
            raise ValueError("Empty response")

        # remove markdown wrappers
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        # try direct parse
        try:
            return json.loads(text)
        except:
            pass

        # fallback: extract largest JSON object
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))

        raise ValueError("No valid JSON found")

    # ---------------------------
    # Confidence Calibration
    # ---------------------------
    def _compute_confidence(self, result):
        score = 0.5

        if result.get("likely_files"):
            score += 0.2

        if result.get("fix_strategy") and len(result["fix_strategy"]) > 40:
            score += 0.2

        if "function" in result.get("reasoning", "").lower():
            score += 0.1

        return round(min(score, 0.95), 2)

    # ---------------------------
    # Issue Structuring
    # ---------------------------
    def _structure_issue(self, title, body):
        body = body[:600]

        return f"""
Problem:
{title}

Details:
{body}

Focus on:
- failure behavior
- affected logic
- likely cause
"""

    # ---------------------------
    # Main Analysis
    # ---------------------------
    def analyze(self, issue_title, issue_body, repo_context, file_tree):

        if not repo_context:
            return {
                "root_cause_summary": "No relevant code retrieved.",
                "likely_files": [],
                "reasoning": "Phase 3 retrieval returned empty.",
                "fix_strategy": "Increase retrieval depth.",
                "agent_prompt": "",
                "confidence": 0.0
            }

        formatted_context = self._format_context(repo_context)
        structured_issue = self._structure_issue(issue_title, issue_body)

        prompt = f"""
    You are a senior software maintenance engineer.

    Follow a STRICT single chain of reasoning:

    1. Identify the failure behavior
    2. Map it to exact code (file + function/reducer)
    3. Identify the precise logic/condition causing the issue
    4. Explain WHY this logic causes the issue (mechanism)
    5. Propose a concrete fix (include pseudo-code or code-level change)

    Then generate an AGENT PROMPT that another AI can use to fix the bug.

    ------------------------
    AGENT PROMPT REQUIREMENTS:
    ------------------------
    - Must describe the bug clearly
    - Must reference exact file(s) and logic (function/reducer)
    - Must include the exact logic that is wrong
    - Must include the exact change required (pseudo-code allowed)
    - Must be step-by-step
    - Must be directly usable (copy-paste)
    - Must NOT include meta text ("you are an AI")
    - Must include expected outcome

    ------------------------
    STRICT RULES:
    ------------------------
    - ONLY use provided repository context
    - DO NOT invent files or logic
    - ONLY include directly responsible files
    - Prefer specific functions over general files
    - Identify exact faulty condition or logic
    - Do NOT be vague

    ------------------------
    ISSUE:
    {structured_issue}

    ------------------------
    REPOSITORY CONTEXT:
    {formatted_context}

    ------------------------
    Strictly Return ONLY valid JSON. Do NOT wrap in markdown. Do NOT add explanation.
    {{
    "root_cause_summary": "...",
    "likely_files": ["file_path"],
    "reasoning": "Step-by-step chain: failure → code → faulty logic → why",
    "fix_strategy": "Include exact code-level or pseudo-code fix",
    "agent_prompt": "...",
    "confidence": 0.0
    }}
    """

        # ---------------------------
        # Token Count
        # ---------------------------
        try:
            token_data = self.llm.count_tokens("gemini", self.model_name, prompt)
            input_tokens = token_data["input_tokens"]
            print(f"[LLM] Input tokens: {input_tokens}")
        except Exception:
            input_tokens = None


        # ---------------------------
        # LLM Call
        # ---------------------------
        result_obj = self.llm.generate("gemini", self.model_name, prompt)

        response_text = result_obj["text"]

        # extract output tokens safely
        try:
            output_tokens = result_obj.get("usage", None)
        except:
            output_tokens = None

        try:
            result = self._safe_json_parse(response_text)
            # ---------------------------
            # Validate agent prompt quality
            # ---------------------------
            agent_prompt = result.get("agent_prompt", "")

            if not agent_prompt or len(agent_prompt) < 80:
                result["agent_prompt"] = f"""
    Fix a bug in the repository.

    ISSUE:
    {issue_title}

    ROOT CAUSE:
    {result.get("root_cause_summary", "")}

    AFFECTED FILES:
    {", ".join(result.get("likely_files", []))}

    PROBLEM:
    The current logic is incorrect and causing unintended behavior.

    REQUIRED FIX:
    {result.get("fix_strategy", "")}

    INSTRUCTIONS:
    1. Locate the relevant function/reducer in the specified file(s)
    2. Identify the faulty logic
    3. Modify the logic to restrict behavior to the correct scope
    4. Ensure no unnecessary side effects
    5. Validate behavior against expected outcome

    EXPECTED RESULT:
    Bug should be fixed and behavior should match intended design.
    """.strip()

            # ---------------------------
            # Auto-calibrate confidence
            # ---------------------------
            result["confidence"] = self._compute_confidence(result)
            result['llmoutput'] = result
            if input_tokens:
                result["input_tokens"] = input_tokens

            if output_tokens:
                result["output_tokens"] = output_tokens
            return result

        except Exception:
            return {
                "root_cause_summary": "Failed to parse model output, raw response preserved.",
                "likely_files": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning": response_text[:2000],  # keep more for debugging
                "fix_strategy": response_text[:2000],  # 🔥 FULL DEBUG COPY
                "agent_prompt": "",
                "confidence": 0.0,
                "llmoutput": response_text
            }
        except Exception as e:
            return {
                "root_cause_summary": "LLM failure.",
                "likely_files": [],
                "input_tokens": input_tokens,
                "reasoning": str(e),
                "fix_strategy": "Model or API failure.",
                "agent_prompt": "",
                "confidence": 0.0,
                "llmoutput": response_text
            }