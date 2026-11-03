import re

# A placeholder is `{name}`. Other braces, such as the JSON example in
# prompts/verification/consistency_v1.txt, are left untouched.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PromptLoader:
    """Utility for loading and filling prompt templates."""

    @staticmethod
    def load_prompt_template(filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Fills `{name}` placeholders. Unlike str.format, literal braces need no escaping."""
        missing = sorted({name for name in PLACEHOLDER.findall(template) if name not in kwargs})
        if missing:
            raise KeyError(f"Prompt template needs values for: {', '.join(missing)}")
        return PLACEHOLDER.sub(lambda match: str(kwargs[match.group(1)]), template)
