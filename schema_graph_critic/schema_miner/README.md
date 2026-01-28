# Schema Miner

A specialized tool for building large-scale datasets of JSON Schemas for training Graph Neural Networks (GNNs). This tool "mines" schemas from two distinct sources: curated high-value repositories and "wild" usage on GitHub.

## Features

*   **Repository Cloning**: Automatically aggregates huge collections of production-grade schemas from Kubernetes, AWS, Azure, SchemaStore, and more.
*   **Wild Mining**: Uses the GitHub Code Search API to find real-world schemas, including those with "messy" definitions or interesting structures (perfect for robust model training).
*   **Deduplication**: Automatically deduplicates downloaded files using MD5 hashing.
*   **Validation**: Performs basic validation to ensure downloaded files are valid JSON.

## Prerequisites

*   Python 3.x
*   Git installed and in your PATH

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Configuration**
    To use the "Wild Mining" feature, you must provide a GitHub Personal Access Token.
    
    Create a `.env` file in the root directory:
    ```bash
    GITHUB_TOKEN=your_token_starts_with_ghp_...
    ```

## Usage

The tool is run via the `miner.py` script. You can run one or both modes.

### 1. Clone High-Value Repositories
This mode downloads curated datasets known for complex graph structures (e.g., Kubernetes, CloudFormation, OpenAPI specs).

```bash
python miner.py --clone
```

**Target Repositories:**
*   `json-schema-org/JSON-Schema-Test-Suite` (Ground truth for correctness)
*   `SchemaStore/schemastore` (Common config schemas)
*   `instrumenta/kubernetes-json-schema` (Deeply nested graphs)
*   `aws/cloudformation-resource-schema` & `Azure/azure-resource-manager-schemas`
*   `APIs-guru/openapi-directory` (OpenAPI/Swagger styles)
*   `HumanCellAtlas/metadata-schema` (Scientific metadata)
*   `CycloneDX/specification` (Recursive SBOM structures)

### 2. Mine the "Wild"
This mode searches GitHub for files matching specific schema patterns (e.g., files containing `$schema` or deeply nested `$defs`).

```bash
# Mine 100 files per search query
python miner.py --search --limit 100
```

*Note: The GitHub Code Search API has a rate limit (approx. 30 requests/minute). The script handles basic errors but large crawls may take time.*

## Output Structure

Data is saved to the `mined_schemas/` directory:

```text
mined_schemas/
├── repositories/       # Cloned git repos (Clean data)
│   ├── kubernetes-json-schema/
│   ├── schemastore/
│   └── ...
└── wild_schemas/       # Individual JSON files from GitHub (Wild data)
    ├── owner_repo_path_to_file.json
    └── ...
```
