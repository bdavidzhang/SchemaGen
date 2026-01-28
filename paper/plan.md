# Plan for writing Schema Gen paper 

We are writing a paper for ICML 2026! We have been building schema bench, a benchmark that tests the LLM's ability to generate JSON schemas for certain scenarios, and schema graph critic, a middleware tool that improves schema generation quality at inference time. 

Files for reference / of interest:
- plan.md (current): plan for writing the paper
- contents.md: planned contents of the paper, in markdown
- abstract.md: abstract for the paper
- main.tex: main tex file for paper
- main.bib: main bibtex file for paper

This will provide information for the paper:
- ../schema_bench/README.md: documentation for schema bench
- ../schema_graph_critic/README.md: documentation for schema graph critic
- ../schema_graph_critic/training.md: documentation for schema graph critic training
- ../schema_graph_critic/high_level_plan.md: high level plan for schema graph 
- ../schema_graph_critic/results/: experiment results (JSON files)

## Completed TODOs:
- [x] Create the paper outline in contents.md
- [x] Based on the work we have so far, write up the planned paper in contents.md
- [x] Write out abstract.md as an abstract for our proposed paper
- [x] Type up the actual paper in main.tex
- [x] Run SchemaGNN evaluation experiments
- [x] Run baseline comparison experiments  
- [x] Run ablation studies (HGT vs GCN vs GAT)
- [x] Update paper with experimental results

## Key Results Summary:
| Experiment | Key Finding |
|------------|-------------|
| SchemaGNN vs Baseline | **84.0% F1** vs 80.2% F1 (+3.8% improvement) |
| Per-corruption detection | 100% precision, 84-99% recall across all types |
| Ablation (HGT vs GCN) | HGT 82.6% F1 vs GCN 76.7% F1 (+5.9% from heterogeneous modeling) |

## Remaining TODOs:
- [ ] Run SchemaBench evaluation on LLMs (GPT-4, Claude, Llama)
- [ ] Run end-to-end LLM pipeline experiment (requires API keys)
- [ ] Address ICML reviewer feedback (see feedback/feedback_1.md)
- [ ] Final paper polish and submission