Hello!

You requested a review of your paper submitted to ICML using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will not be used in the review process. Reviewers, area chairs, and program committee members will not have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.

PAT Feedback Model: MODEL_A

PATLibraryRunPipeline:

HIGH LEVEL SUMMARY
Paper Summary
The paper, "SchemaGen: Neuro-Symbolic JSON Schema Generation," addresses the challenge of evaluating and improving the capability of Large Language Models (LLMs) to design JSON Schemas (data contracts). The authors introduce two main contributions. First, SchemaBench, a benchmark comprising 40 scenarios designed to evaluate LLMs as both schema designers ("architects") and instance generators ("builders") across varying complexity, constraints, and ambiguity. Second, SchemaGraph Critic, a neuro-symbolic middleware that represents JSON Schemas as heterogeneous graphs and employs a Graph Neural Network (SchemaGNN, based on HGT) to detect structural logic errors (e.g., dangling references, constraint conflicts) that standard validators may miss. This critic is integrated into a feedback loop to iteratively refine LLM-generated schemas. The SchemaGNN is trained on synthetically corrupted data. Experimental results are pending.

Key Issues Roadmap
1. Introduction, Motivation, and Related Work: Please clarify the scope of "semantic correctness" validation, ensuring consistency between the claims in the introduction/abstract and the limitations acknowledged in Section 6.2 regarding the distinction between structural logic and semantic alignment with requirements.
2. SchemaBench: Benchmark Design and Metrics: Address the underspecification of evaluation metrics (Specificity Score, Key Coverage) and the missing methodology for calculating Gate 3 "Pass Rates" from continuous metrics. Additionally, review the potential limitation that the Constraint Recall metric evaluates keyword presence rather than the correctness of the associated values.
3. SchemaGraph Critic: Neuro-Symbolic Methodology: Provide essential methodological details required for reproducibility, specifically regarding the graphification algorithm, the specifics of node feature engineering (e.g., input for semantic embeddings), and the procedures for synthetic data corruption and ground truth localization.
4. Experimental Setup and Analysis: Introduce necessary comparative baselines for the experimental design, including alternative refinement strategies (e.g., LLM self-correction) for the end-to-end evaluation, and deterministic graph algorithms for the SchemaGNN evaluation. Additionally, please verify the claim in Section 6.1 regarding the behavior of standard validators when handling unresolvable references.
DETAILED SEGMENT REVIEWS
[1] SEGMENT: 1. Introduction, Motivation, and Related Work
PAGES: [[1, 2]]
1. Summary

The reviewed segment (Sections 1, 2, and motivation in 3.1 and 4.1) introduces the research objective: evaluating and improving the capability of Large Language Models (LLMs) to design JSON Schemas. Section 1 defines the problem space, arguing that the ability of LLMs to act as "architects" (designing data contracts) is underexplored compared to their ability to act as "builders" (generating compliant data). It identifies specific challenges: long-distance dependencies, structural logic, and recursion. The section outlines the two main contributions: SchemaBench (a benchmark) and SchemaGraph Critic (a GNN-based validation middleware). Section 2 (Related Work) positions the work against constrained decoding and standard validators, arguing these existing methods focus on syntactic validity rather than structural logic. Sections 3.1 and 4.1 provide specific motivations, emphasizing the need for evaluation benchmarks focused on schema design and the utility of GNNs for identifying topological issues that sequential models struggle with.

2. Potential Mistakes and Improvements

Clarity (Consistency in Terminology): There is a potential inconsistency in the terminology used to describe the validation capabilities across the paper. The Abstract (L029) states the approach "validates semantic correctness." Similarly, Section 2.1 emphasizes that existing approaches guarantee syntactic validity but not "semantic correctness." However, Section 6.2 (Limitations, L214-216) explicitly states: "The GNN detects structural errors, not semantic misalignment with requirements." The introduction and related work should clarify the scope of "semantic correctness" as used in this context, clearly distinguishing between the internal structural logic (which the GNN validates) and the alignment with the original natural language requirements (which it does not).
Clarity (Technical Precision): In Section 1 (L046-048), the text describes long-distance dependencies created by $ref as "creating dependencies invisible to autoregressive models." This phrasing is imprecise for Transformer-based LLMs. Dependencies within the context window are visible via the attention mechanism. The challenge lies not in visibility, but in the difficulty of reliably reasoning about the non-sequential, graph-like structure implied by these references when processed autoregressively. The authors might consider revising the phrasing to state that these dependencies are challenging for autoregressive models to resolve structurally.
3. Minor Corrections and Typos

None identified.
[2] SEGMENT: 2. SchemaBench: Benchmark Design and Metrics
PAGES: [[2, 2], [6, 6]]
Summary
The reviewed segment (Section 3 and Appendix A) introduces SchemaBench, a benchmark designed to evaluate Large Language Models (LLMs) in two capacities: as "Architects" (designing JSON Schemas from requirements) and "Builders" (generating conforming instances). The benchmark comprises 40 scenarios organized into three tracks: Structural complexity, Constraint hardness, and Ambiguity resolution (Sections 3.2, 3.4). The evaluation uses a three-gate pipeline (Section 3.3): Gate 1 (Syntax & Meta-Validity), Gate 2 (Self-Consistency), and Gate 3 (Semantic Alignment). Gate 3 employs three metrics: Constraint Recall (CR, Eq. 1), Specificity Score, and Key Coverage. Appendix A provides examples of the benchmark scenarios.

Potential Mistakes and Improvements
The following points concern the validity, clarity, and reproducibility of the benchmark design and evaluation metrics defined in Section 3.

Limitations in Evaluating Semantic Alignment (Validity) (Section 3.3): The authors state the goal is to evaluate the generation of "semantically correct" schemas (L079). However, the primary metric for Semantic Alignment, Constraint Recall (CR, Eq. 1), only evaluates the presence of expected JSON Schema keywords (e.g., minimum, pattern), based on the required_constraints defined in Section 3.2.2. This metric does not validate the correctness of the associated values against the input requirements. For example, if the requirement specifies a numerical bound, CR confirms the usage of the minimum keyword, but the pipeline does not describe a mechanism for verifying that the numerical value implemented in the schema is correct. This limits the evaluation of true semantic alignment with the requirements.

Underspecified Evaluation Metrics (Clarity) (Section 3.3): Two metrics in Gate 3 lack the formal definitions required for a reproducible benchmark:

Specificity Score: Defined as the "Ratio of constrained fields to total fields." The criteria for what constitutes a "constrained field" are ambiguous (e.g., whether specifying type is sufficient, or if validation keywords like minLength are required). Additionally, the method for enumerating "total fields" and "constrained fields" within complex, nested, or polymorphic schemas (e.g., involving oneOf or $ref) is not defined.
Key Coverage: Defined qualitatively as "Does the schema define expected properties?". A quantitative formulation (e.g., Recall or F1 score relative to the gold_keys mentioned in Section 3.2.2) is necessary.
Missing Methodology for Pass Rate Calculation (Clarity) (Section 3.3): The evaluation pipeline defines continuous metrics for Gate 3 (e.g., CR is a ratio). However, the results presented in Table 7 report "Pass Rates (%)" per gate. The methodology in Section 3.3 does not specify how these continuous metrics are thresholded or aggregated to determine a binary pass/fail outcome for Gate 3.

Evaluation Methodology for Ambiguity (Clarity/Validity) (Sections 3.2.1, 3.3): Track C is designed to evaluate the resolution of ambiguity, such as inferring "reasonable defaults" or "implied fields" (L096). However, the evaluation metrics rely on predefined required_constraints and gold_keys (Section 3.2.2). This reliance on a fixed ground truth seems contradictory to evaluating ambiguous scenarios where multiple valid interpretations may exist. The methodology does not specify how the "reasonableness" of the LLM's interpretation is objectively assessed against a singular ground truth.

Scope of "Builder" Evaluation (Validity) (Section 3.3): The "Builder" role (L081) is evaluated solely through Gate 2 (Self-Consistency), which checks if the generated instance validates against the generated schema. This approach does not assess the completeness or diversity of the generated instance. An LLM could potentially pass this gate by generating a minimal instance (e.g., including only required fields) without demonstrating the capability to generate comprehensive instances that cover various aspects of the schema (e.g., optional fields, different polymorphic branches).

Benchmark Construction Details (Clarity) (Section 3.4): The paper introduces a new benchmark comprising 40 scenarios but does not describe the methodology for constructing this dataset. Details regarding how these scenarios were sourced (e.g., manually created, adapted from real-world repositories), curated, and validated for coverage of the JSON Schema specification are necessary to assess the benchmark's comprehensiveness and representativeness.

Minor Corrections and Typos
None identified in this segment.
[3] SEGMENT: 3. SchemaGraph Critic: Neuro-Symbolic Methodology
PAGES: [[2, 4], [6, 6]]
1. Summary The reviewed segment (Section 4, Appendices B and C) introduces the SchemaGraph Critic, a neuro-symbolic middleware designed to identify structural logic errors in JSON Schemas that standard validators miss. The methodology operates in a five-step feedback loop (Section 4.2). Central to this is the conversion of JSON Schemas into a heterogeneous graph representation (Section 4.3) with defined node types, edge types, and features. A Heterogeneous Graph Transformer (HGT), named SchemaGNN (Section 4.4), processes this graph to predict overall validity (Global Critic Head) and localize errors (Local Debugger Head). The model is trained using synthetically corrupted data (Section 4.5) generated by "The Corruptor" (Table 6) and optimized with a joint loss function (Section 4.6). Finally, GNN outputs are translated into natural language feedback for LLM refinement (Section 4.7, Appendix C).

2. Potential Mistakes and Improvements

Clarity (Graphification Algorithm - Sections 4.2 and 4.3): The paper defines the structure of the heterogeneous graph representation (Tables 3 and 4). However, the algorithmic details of the "Graphification" process (Section 4.2, Step 2)—how a JSON Schema is parsed and transformed into this graph structure—are omitted. The rules for mapping complex schema constructs, such as logical combinators (allOf, anyOf) or conditional subschemas (if/then), to the defined nodes and edges are not specified. This omission hinders the reproducibility of the input representation.
Clarity (Node Feature Engineering - Section 4.3.3): The description of the node features (Table 5) lacks specificity required for reimplementation:
Semantic Embedding: It is unclear what text input is used to generate the 384-dimensional "Semantic embedding" (MiniLM-L6). It is not specified if the input is the schema keyword, the property name, associated metadata (e.g., description), or the JSON path.
Constraint Flags: The paper mentions 8 dimensions for "Constraint flags" but does not enumerate which specific JSON Schema constraints (e.g., minimum, pattern) these flags represent.
Clarity (Training Data Synthesis and Ground Truth Labeling - Section 4.5): The description of the synthetic training data generation is insufficient for reproducibility.
The algorithmic implementation details for introducing the 7 corruption types (Table 6) are omitted (e.g., how a CONSTRAINT_CONFLICT is systematically generated).
The methodology does not explain how the ground truth localization (identifying the specific nodes 
 that are the root cause of the error) is established during the corruption process. This ground truth is necessary to train the Local Debugger Head (Eq. 3) and calculate the local loss component (Eq. 4).
Clarity (Feedback Translation Mechanism - Section 4.7): Section 4.7 describes translating GNN outputs to natural language. Step 1 mentions mapping node IDs to JSON paths via "stored metadata." The mechanism for generating and storing this metadata during the graphification process is not explained. Furthermore, the feedback relies on generic templates based on node types (Appendix C). It is unclear how the system incorporates specific context about the detected error (e.g., the actual values of conflicting constraints or the name of a missing definition) into the feedback, which is necessary for providing actionable guidance to the LLM.
3. Minor Corrections and Typos

Page 4, Line 177: In the description following Table 6, the notation for the predicted error probability appears as 
 ("
s the predicted..."). This should likely be 
.
[4] SEGMENT: 4. Experimental Setup and Analysis
PAGES: [[4, 5]]
1. Summary

The reviewed segment comprises Sections 5 (Experiments) and 6 (Analysis and Discussion). Section 5 outlines the experimental setup, including the Large Language Models (LLMs) selected for evaluation on SchemaBench (e.g., GPT-4-Turbo, Llama 3.1, Claude 3.5 Sonnet) and the training details for the SchemaGNN (data sources, hardware, training time). Sections 5.2 and 5.3 present placeholder tables (Tables 7 and 8), noting that results are pending. Section 5.4 describes the protocol for evaluating the end-to-end pipeline, aiming to measure the improvement gained from iterative refinement using the SchemaGraph Critic. Section 6.1 provides an analysis, using an example in Table 9, arguing why GNNs outperform standard validators for structural logic validation. Section 6.2 discusses the limitations of the proposed approach.

2. Potential Mistakes and Improvements

The review focuses on the rigor of the experimental design and the validity of the analysis, given that results are pending.

Experimental Validity: Baselines

Lack of Comparative Baselines for Iterative Refinement (Section 5.4). The end-to-end evaluation protocol measures improvement from the initial LLM generation (Round 0) to the refined output (Round N) using the SchemaGraph Critic. However, the experimental design lacks comparative baselines necessary to isolate the contribution of the GNN-based feedback. To validate the efficacy of the proposed critic, the evaluation should include comparisons against alternative refinement strategies, such as (a) LLM self-correction without external feedback, or (b) refinement using feedback solely from a standard JSON Schema validator. Without these comparisons, it cannot be determined if the improvement stems from the GNN's specific insights or merely from the iterative process.
Lack of Baselines for SchemaGNN (Section 5.3). The evaluation plans to measure the SchemaGNN's performance in defect detection (Table 8). However, several corruption types listed in Table 6 (e.g., DANGLING_REF, CIRCULAR_REF) are detectable using deterministic graph algorithms (e.g., reference resolution checks, cycle detection). The evaluation should include these symbolic methods as baselines to quantify the advantage of a learned GNN approach over existing deterministic methods.
Experimental Validity: Generalization and Data

Generalization from Synthetic Data (Sections 5.1, 5.3). The SchemaGNN is trained using synthetically generated corruptions (Section 4.5). The composition of the test set used to evaluate the GNN performance (Table 8) is not specified. If the evaluation relies solely on held-out synthetic data, it may overestimate the model's performance on authentic structural errors generated by LLMs (a limitation acknowledged in Section 6.2.3). The authors should clarify the test set composition and ideally evaluate on a dataset of actual LLM-generated schemas.
Correctness of Analysis

Validator Behavior Example (Section 6.1). Table 9 illustrates the advantage of the GNN using an example of a dangling reference ($ref to a non-existent node). It claims that a standard validator (e.g., ajv) will pass (✓) this schema because it is "Syntactically correct." This claim requires verification or clarification, as many standard JSON Schema validators attempt to resolve references during schema compilation and will report an error if a reference is unresolvable. The authors should clarify the specific validation context or utilize an example that standard validators inherently ignore (e.g., the constraint conflict mentioned in Section 1).
Clarity and Reproducibility

Insufficient Detail on Training Data Synthesis (Section 5.1). The methodology for creating the SchemaGNN training dataset is underspecified. For reproducibility, the authors should report the distribution of the 7 corruption types (Table 6) and the ratio of valid to invalid schemas in the training set.
Underspecified Evaluation Protocols (Sections 5.1, 5.4). Key details are missing from the experimental setup:
Section 5.1 lists the LLMs evaluated but omits configuration details such as temperature, decoding strategies, and prompting techniques (e.g., zero-shot vs. few-shot).
Section 5.4 describes an iterative protocol (Round 1–N) but does not specify the maximum value of N or the stopping criteria for the loop. It also does not specify the metrics used in Step 4 ("Measure improvement").
Ablation and Attribution

Lack of Ablation Studies (Section 5.3). The experimental design does not mention ablation studies to justify the architectural choices of the SchemaGNN. To analyze the contributions of different components, experiments isolating the impact of the Heterogeneous Graph Transformer (e.g., compared to a homogeneous GNN) and the engineered node features (e.g., the impact of semantic embeddings from Table 5) would be necessary.
3. Minor Corrections and Typos

Page 4, Line 200: "Llama 3.1 (70B, 405B" is missing a closing parenthesis.
