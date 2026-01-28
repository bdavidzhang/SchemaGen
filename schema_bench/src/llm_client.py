import os
import json
from openai import OpenAI
from .config_loader import LLMConfig

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env_var)
        self.client = None
        
        if config.provider == "openai":
            if not self.api_key:
                print(f"WARNING: {config.api_key_env_var} not found in environment variables. Using Mock mode.")
            else:
                self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def generate(self, prompt: str) -> tuple[str, str]:
        """
        Generates schema and instance from the prompt.
        Returns (schema_str, instance_str).
        """
        if not self.client:
            return self._mock_generate(prompt)

        system_prompt = (
            "You are an expert Data Engineer. Your task is to generate two artifacts based on the user's request:\n"
            "1. A valid JSON Schema (Draft 2020-12).\n"
            "2. A valid JSON Instance that strictly adheres to that schema.\n"
            "\n"
            "Return the output in the following JSON format:\n"
            "{\n"
            "  \"schema\": { ... your schema object ... },\n"
            "  \"instance\": { ... your instance object ... }\n"
            "}\n"
            "Do not include markdown formatting (like ```json), just the raw JSON object."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                response_format={"type": "json_object"} # Force JSON output if supported
            )
            
            content = response.choices[0].message.content
            # Parse the wrapper JSON
            try:
                data = json.loads(content)
                schema_str = json.dumps(data.get("schema", {}))
                instance_str = json.dumps(data.get("instance", {}))
                return schema_str, instance_str
            except json.JSONDecodeError:
                # Fallback if model failed to produce valid JSON wrapper
                # For now, just return empty to signal failure
                return "{}", "{}"
                
        except Exception as e:
            print(f"LLM API Error: {e}")
            return "{}", "{}"

    def _mock_generate(self, prompt: str) -> tuple[str, str]:
        # Fallback Mock implementation for testing without API key
        print(" [Mock Mode Triggered] ")
        if "web server configuration" in prompt:
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
                    "host": {"type": "string"},
                    "debug_mode": {"type": "boolean"}
                },
                "required": ["port", "host", "debug_mode"]
            }
            instance = {"port": 8080, "host": "localhost", "debug_mode": True}
            return json.dumps(schema), json.dumps(instance)
        
        return "{}", "{}"
