# Abstract

Large Language Models (LLMs) have become powerful generators of structured data, yet their ability to **design** data contracts—rather than merely comply with them—remains underexplored. We introduce **SchemaBench**, the first benchmark evaluating LLMs as schema architects, testing their capacity to create valid, specific, and semantically correct JSON Schemas across varying structural complexity, constraint hardness, and ambiguity levels. Complementing this, we present **SchemaGraph Critic**, a neuro-symbolic middleware that represents JSON Schemas as heterogeneous graphs and employs a Graph Neural Network (specifically, a Heterogeneous Graph Transformer) to detect structural logic errors invisible to standard validators—such as dangling references, circular dependencies, and constraint conflicts. Unlike constrained decoding, which ensures syntactic validity, our approach validates semantic correctness and generates natural language feedback for iterative LLM refinement. Our experiments demonstrate that current state-of-the-art models struggle with recursive and polymorphic schema design, achieving only [X]% self-consistency on complex scenarios, while SchemaGraph Critic achieves [Y]% accuracy in detecting structural defects and improves schema quality by [Z]% through its feedback loop. Together, SchemaBench and SchemaGraph Critic establish a rigorous framework for evaluating and improving LLM-generated data contracts.

# Primary Area

Neurosymbolic AI; Applications (Natural Language Processing)

# Keywords

JSON Schema, structured output generation, graph neural networks, heterogeneous graph transformer, LLM evaluation, benchmark, neuro-symbolic AI, schema validation, data contracts
