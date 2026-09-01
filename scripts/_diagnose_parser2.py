"""Diagnostic script 2 for judge response parser validation."""
import re
import json
from pathlib import Path
from src.evaluation.judge import _extract_json_object, _strip_markdown_fence

def test_parse(text: str) -> dict:
    res = {}
    try:
        json.loads(text.strip())
        res["direct"] = True
    except Exception as e:
        res["direct"] = False

    fenced = _strip_markdown_fence(text)
    if fenced:
        try:
            json.loads(fenced)
            res["fenced"] = True
        except Exception:
            res["fenced"] = False
    else:
        res["fenced"] = None

    raw_for_ext = fenced if fenced else text
    candidate = _extract_json_object(raw_for_ext)
    if candidate:
        try:
            json.loads(candidate)
            res["brace_balanced"] = True
        except Exception as e:
            res["brace_balanced"] = False
            res["bb_err"] = str(e)
    else:
        res["brace_balanced"] = None

    m = re.search(r"\{[^{}]*\}", raw_for_ext, re.DOTALL)
    if m:
        try:
            json.loads(m.group(0))
            res["old_regex"] = True
        except Exception as e:
            res["old_regex"] = False
            res["regex_err"] = str(e)
    else:
        res["old_regex"] = None

    return res

if __name__ == "__main__":
    print("Diagnostics 2 ready")
