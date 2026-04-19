# skills/prompts.py
# ============================================================
# Prompt templates for LLM-based skill extraction.
# Keeping prompts in a dedicated module makes them easy to
# iterate on without touching business logic.
# ============================================================

SKILL_EXTRACTION_SYSTEM = """\
You are an expert academic curriculum analyst.
Your job is to read fragments of university course syllabi and extract
structured information about the skills, topics, and knowledge areas covered.

RULES:
- Always respond with VALID JSON only. No prose, no markdown fences.
- Skills must be specific (e.g. "linear regression" not "math").
- Difficulty: "beginner", "intermediate", or "advanced".
- Domain: the primary academic/professional field.
- If you cannot determine a field, use "general".
- List 3-10 skills per chunk. Be concise and precise.
"""

SKILL_EXTRACTION_PROMPT = """\
Analyze the following university syllabus fragment and extract structured skill information.

SYLLABUS FRAGMENT:
\"\"\"
{chunk_text}
\"\"\"

Respond ONLY with a JSON object matching this exact schema:
{{
  "skills": ["skill_1", "skill_2", "skill_3"],
  "difficulty": "beginner" | "intermediate" | "advanced",
  "domain": "string (e.g. machine learning, calculus, software engineering)",
  "topics": ["topic_1", "topic_2"],
  "prerequisites": ["prereq_1"]
}}
"""

# Prompt for document-level skill aggregation (called once per full document)
DOCUMENT_SKILL_AGGREGATION_PROMPT = """\
Given the following list of per-chunk skills extracted from a single course syllabus,
produce a unified skill profile for the entire course.

PER-CHUNK SKILLS (JSON list):
{chunk_skills_json}

Respond ONLY with a JSON object:
{{
  "skills": ["top skill_1", "skill_2", ...],
  "difficulty": "beginner" | "intermediate" | "advanced",
  "domain": "primary domain string",
  "topics": ["major topic_1", ...],
  "prerequisites": ["prereq_1", ...]
}}
"""
