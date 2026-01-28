import os
import subprocess
import argparse
import hashlib
import json
from github import Github
from dotenv import load_dotenv
from jsonschema import (
    Draft4Validator,
    Draft6Validator,
    Draft7Validator,
    Draft202012Validator,
    SchemaError,
)

# Load environment variables from .env file
load_dotenv()

# List of high-value repositories to clone
REPOS_TO_CLONE = [
    "https://github.com/json-schema-org/JSON-Schema-Test-Suite.git",
    "https://github.com/SchemaStore/schemastore.git",
    "https://github.com/instrumenta/kubernetes-json-schema.git",
    "https://github.com/aws/cloudformation-resource-schema.git",
    "https://github.com/Azure/azure-resource-manager-schemas.git",
    "https://github.com/APIs-guru/openapi-directory.git",
    "https://github.com/HumanCellAtlas/metadata-schema.git",
    "https://github.com/CycloneDX/specification.git"
]

DATA_DIR = "mined_schemas"

def clone_repos():
    """Clones the curated list of repositories."""
    repos_dir = os.path.join(DATA_DIR, "repositories")
    if not os.path.exists(repos_dir):
        os.makedirs(repos_dir)
    
    print(f"--- Cloning {len(REPOS_TO_CLONE)} Repositories ---")
    for repo_url in REPOS_TO_CLONE:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(repos_dir, repo_name)
        
        if os.path.exists(repo_path):
            print(f"Skipping {repo_name} (already exists).") 
        else:
            print(f"Cloning {repo_name}...")
            try:
                subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone {repo_name}: {e}")

def get_file_hash(content):
    """Returns MD5 hash of content for deduplication."""
    return hashlib.md5(content).hexdigest()

# File to persist seen hashes across runs
HASHES_FILE = os.path.join(DATA_DIR, ".seen_hashes.json")

def load_seen_hashes():
    """Load previously seen hashes from disk."""
    if os.path.exists(HASHES_FILE):
        try:
            with open(HASHES_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()

def save_seen_hashes(hashes):
    """Save seen hashes to disk."""
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HASHES_FILE, "w") as f:
        json.dump(list(hashes), f)

# Validators to try, in order from newest to oldest
# We try multiple drafts since schemas in the wild may use different versions
SCHEMA_VALIDATORS = [
    Draft202012Validator,
    Draft7Validator,
    Draft6Validator,
    Draft4Validator,
]

def is_json_schema(data):
    """
    Validates if a parsed JSON object is a valid JSON Schema.
    
    Uses the jsonschema library to validate against the official meta-schema.
    Tries multiple draft versions (2020-12, 7, 6, 4) to be lenient with
    schemas that may use different versions.
    
    Returns True if the data is a valid JSON Schema according to any draft.
    """
    if not isinstance(data, dict):
        return False
    
    # Try each validator - if any accepts it, it's a valid schema
    for validator_cls in SCHEMA_VALIDATORS:
        try:
            validator_cls.check_schema(data)
            return True
        except SchemaError:
            # This draft doesn't accept it, try the next one
            continue
    
    # None of the validators accepted it
    return False

def search_github(token, limit=50):
    """Searches GitHub for schemas using defined strategies."""
    g = Github(token)
    
    # Queries from the plan - use extension:json to strictly match .json files only
    queries = [
        'extension:json "$schema": "http://json-schema.org"', # Strategy A: Standard Header
        'extension:json "definitions": AND "$ref":',          # Strategy B: Definitions Hunter
        'extension:json filename:.schema'                     # Strategy C: Config File Approach
    ]
    
    wild_dir = os.path.join(DATA_DIR, "wild_schemas")
    if not os.path.exists(wild_dir):
        os.makedirs(wild_dir)
        
    # Load hashes from persistent storage
    seen_hashes = load_seen_hashes()
    initial_hash_count = len(seen_hashes)
    
    if initial_hash_count > 0:
        print(f"Loaded {initial_hash_count} existing file hashes for deduplication.")
    
    print(f"--- Searching GitHub (Limit: {limit} per query) ---")

    for query in queries:
        print(f"Query: {query}")
        try:
            result = g.search_code(query)
            count = 0
            
            # Note: totalCount might be inaccurate for code search, just iterating
            for repo_file in result:
                if count >= limit:
                    break
                
                try:
                    # Skip files that don't have .json extension
                    if not repo_file.path.lower().endswith('.json'):
                        continue
                    
                    content = repo_file.decoded_content
                    
                    # 1. Validation: check if it's valid JSON and looks like a JSON Schema
                    try:
                        json_content = json.loads(content)
                    except json.JSONDecodeError:
                        # Skip invalid JSON files
                        continue
                    
                    if not is_json_schema(json_content):
                        # Skip JSON files that don't look like JSON Schemas
                        continue

                    # 2. Deduplication
                    file_hash = get_file_hash(content)
                    if file_hash in seen_hashes:
                        continue
                    seen_hashes.add(file_hash)
                    
                    # 3. Save
                    # Construct a safe filename: Owner_Repo_Path
                    safe_name = f"{repo_file.repository.full_name.replace('/', '_')}__{repo_file.path.replace('/', '_')}"
                    if len(safe_name) > 200: # Truncate if too long
                        safe_name = safe_name[:200]
                    
                    file_path = os.path.join(wild_dir, safe_name)
                    
                    # Skip if file already exists on disk
                    if os.path.exists(file_path):
                        print(f"Skipping (already exists): {safe_name}")
                        continue
                    
                    with open(file_path, "wb") as f:
                        f.write(content)
                    
                    print(f"Downloaded: {safe_name}")
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing file {repo_file.path}: {e}")
                    
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")
            print("Note: GitHub Code Search API has rate limits (30 req/min).")
    
    # Save updated hashes to disk
    if len(seen_hashes) > initial_hash_count:
        save_seen_hashes(seen_hashes)
        print(f"Saved {len(seen_hashes) - initial_hash_count} new hashes to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Miner Tool")
    parser.add_argument("--clone", action="store_true", help="Clone standard repositories")
    parser.add_argument("--search", action="store_true", help="Search GitHub (requires GITHUB_TOKEN)")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of files per search query")
    
    args = parser.parse_args()
    
    # Create main data dir
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if args.clone:
        clone_repos()
        
    if args.search:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            print("Error: GITHUB_TOKEN environment variable is required for search.")
            print("Please run: export GITHUB_TOKEN=your_token")
        else:
            search_github(token, args.limit)
            
    if not args.clone and not args.search:
        print("Please specify an action:")
        print("  python miner.py --clone   (To download huge repo datasets)")
        print("  python miner.py --search  (To mine individual files from GitHub)")
