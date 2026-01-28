from .models import BenchmarkScenario

SCENARIOS = [
    BenchmarkScenario(
        id="scen_001_flat_config",
        name="Simple App Configuration",
        description="A flat key-value configuration file for a web server.",
        difficulty="Easy",
        track="Structural: Flat",
        prompt_template="Create a JSON schema and a valid JSON instance for a web server configuration. It must include a 'port' (integer between 1024 and 65535), a 'host' (string), and a 'debug_mode' (boolean).",
        required_constraints=["minimum", "maximum", "type"],
        gold_keys=["port", "host", "debug_mode"]
    ),
    BenchmarkScenario(
        id="scen_002_user_profile",
        name="User Profile with Constraints",
        description="A user profile object with nested address and strict validation.",
        difficulty="Medium",
        track="Structural: Nested",
        prompt_template="Create a schema and instance for a User Profile. \n"
                        "Fields required:\n"
                        "- id (uuid format)\n"
                        "- username (string, 3-20 chars)\n"
                        "- email (email format)\n"
                        "- address (nested object with street, city, zipcode)\n"
                        "- roles (array of strings, must be one of 'admin', 'editor', 'viewer')",
        required_constraints=["format", "minLength", "maxLength", "enum", "type"],
        gold_keys=["id", "username", "email", "address", "roles"]
    ),
    BenchmarkScenario(
        id="scen_003_recursive_fs",
        name="Recursive File System",
        description="A recursive directory structure where folders contain files or other folders.",
        difficulty="Recursive",
        track="Structural: Recursive",
        prompt_template="Create a Recursive JSON schema and a valid instance for a File System Node.\n"
                        "A Node must have:\n"
                        "- 'name' (string)\n"
                        "- 'type' (enum: 'file', 'directory')\n"
                        "If type is 'file', it must have 'size' (integer > 0).\n"
                        "If type is 'directory', it must have 'children' (array of Nodes).\n"
                        "The schema must use recursion ($ref).",
        required_constraints=["$ref", "enum", "if", "then", "minimum"], # or OneOf depending on implementation
        gold_keys=["name", "type", "children", "size"]
    ),
    BenchmarkScenario(
        id="scen_004_polymorphic_ui",
        name="Polymorphic UI Components",
        description="A list containing different types of objects with distinct validation rules.",
        difficulty="Hard",
        track="Structural: Polymorphic",
        prompt_template="Create a schema for a 'Page' containing a list of 'components'.\n"
                        "The components array can contain mixed types:\n"
                        "1. Button: { type: 'button', label: string, onClick: string }\n"
                        "2. Image: { type: 'image', src: url string, width: integer }\n"
                        "3. Text: { type: 'text', content: string }\n"
                        "Use 'oneOf' to ensure that a Button cannot have a 'src' and an Image cannot have an 'onClick'.",
        required_constraints=["oneOf", "const", "required"],
        gold_keys=["type", "components", "label", "src", "content"]
    ),
    BenchmarkScenario(
        id="scen_005_strict_regex",
        name="Strict Product SKU",
        description="Testing specific string patterns and format constraints.",
        difficulty="Medium",
        track="Constraint Hardness",
        prompt_template="Create a Product schema.\n"
                        "Fields:\n"
                        "- sku: Must match pattern 'PROD-[A-Z]{3}-[0-9]{3}' (e.g., PROD-ABC-123)\n"
                        "- manufacture_date: standard date format\n"
                        "- server_ip: standard IPv4 format\n"
                        "- tags: Array of unique strings (max 5 items)",
        required_constraints=["pattern", "format", "uniqueItems", "maxItems"],
        gold_keys=["sku", "manufacture_date", "server_ip", "tags"]
    ),
    BenchmarkScenario(
        id="scen_006_ambiguity_event",
        name="Ambiguous User Event",
        description="A vague prompt to test if the model assumes reasonable defaults (timestamp, id).",
        difficulty="Easy",
        track="Ambiguity Resolution",
        prompt_template="Create a schema for a 'user_login' event log. I just need to track that a user logged in.",
        # We expect the model to infer that we need a timestamp and a user identifier, even if not asked.
        required_constraints=[], 
        gold_keys=["timestamp", "user_id", "event_id"]
    )
]
