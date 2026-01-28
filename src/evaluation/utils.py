import json
from typing import Optional, Dict

def parse_model_output_as_json(output: str) -> Optional[Dict]:
    """Parse model output as JSON, handling various formats."""
    if not isinstance(output, str):
        return output if isinstance(output, dict) else None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    return None


def extract_html_table(output: str) -> str:
    """Extract HTML table from model output."""
    if not isinstance(output, str):
        return str(output)

    output = output.strip()

    if "<table" in output.lower():
        start = output.lower().find("<table")
        end = output.lower().find("</table>")
        if end > start:
            return output[start : end + 8]

    if "```html" in output:
        start = output.find("```html") + 7
        end = output.find("```", start)
        if end > start:
            return output[start:end].strip()

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            content = output[start:end].strip()
            if "<table" in content.lower():
                return content

    return output
