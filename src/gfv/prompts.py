class PromptLoader:
    """Utility for loading and filling prompt templates."""
    
    @staticmethod
    def load_prompt_template(filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        return template.format(**kwargs)