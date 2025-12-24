# Task Management Assistant

A small **GraphRAG-style** prototype that answers questions using a **project management knowledge graph** (RDFLib + SPARQL) and a **local Hugging Face LLM** (FLAN‑T5) wrapped for LangChain.

> Key idea: retrieve **structured context** from the graph (not vector embeddings), then force the LLM to answer **only** from that context.

---

## What this notebook does

1. **Builds/loads an RDF knowledge graph** from Turtle strings:
   - An **ontology** (classes/properties like `ex:Project`, `ex:Task`, `ex:Agent`, `ex:Decision`)
   - A **data graph** with example projects, tasks, agents, decisions, statuses, etc.
2. **Queries the graph with SPARQL** to retrieve task-centric context (status, assignee, project, decisions).
3. **Initializes a local LLM** (`google/flan-t5-base`) using `transformers` and wraps it as a LangChain-compatible LLM.
4. **Matches a user question to the best task label** using a simple **token-overlap heuristic**.
5. **Generates an answer** by prompting the LLM with:
   - the retrieved graph context
   - the question
   - strict instruction to answer using **only** the context

---

## How GraphRAG is implemented here

### 1) Task selection (lightweight “retrieval”)
The notebook lists all `ex:Task` labels from the KG, tokenizes them, and picks the label with the **highest overlap** with the question tokens.

### 2) SPARQL context retrieval
Given the selected task label, the notebook runs SPARQL to fetch:
- task IRI + label
- status
- assigned agent
- related project
- any linked decisions

The results are formatted into a clean **context block**.

### 3) Constrained generation
The LLM is prompted to answer **only** from the context and to respond with:

> `"The knowledge graph does not contain this information."`

when the answer is missing.

---

## Requirements

- Python 3.9+ recommended
- Packages:
  - `rdflib`
  - `transformers`
  - `torch`
  - `langchain-community`

> Note: `flan-t5-base` can run on CPU, but GPU will be faster.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate   # (Windows)

pip install -U rdflib transformers torch langchain-community
```

---

## Run

Open and run the notebook:

- `RAG_Assistant.ipynb`

Typical flow inside the notebook:
1. Parse `ONTOLOGY_TTL` and `DATA_TTL` into an RDFLib `Graph`
2. Initialize the FLAN‑T5 pipeline
3. Call `answer_question_with_graph(question, g, llm)` on sample questions

---

## Example questions

The notebook includes examples like:

- “Why is the notifications task blocked?”
- “Who is responsible for building the data pipeline?”
- “What is the status of the task building the data pipeline?”
- “Who is working on the frontend UI?”
- “What are the tasks that Rahaf is in charge of?”

---

## Key functions (where to look)

- `get_task_context(task_label, graph)`  
  SPARQL query + formatting to build the context text passed to the LLM.

- `guess_task_label_from_question(question, graph)`  
  Token-overlap heuristic to select the most relevant task label.

- `answer_question_with_graph(question, graph, llm)`  
  End-to-end: match task → fetch context → prompt LLM → return answer.

---

## Troubleshooting

### “Could not find any task for question …”
The question didn’t match any task label (token overlap = 0).
- Try using words that appear in the task label (e.g., “frontend UI”, “data pipeline”).
- Improve matching by adding:
  - synonyms
  - stemming/lemmatization
  - fuzzy matching (e.g., `rapidfuzz`)

### Slow / high memory usage
Loading `flan-t5-base` can be slow on CPU. Consider:
- using a smaller model (`google/flan-t5-small`)
- running on a GPU

---

## Limitations / Next improvements

- Replace token overlap with better retrieval:
  - TF‑IDF / BM25 over task labels + descriptions
  - embeddings + vector store for hybrid GraphRAG
- Add support for **multi-hop** questions across tasks/projects/agents
- Add evaluation (exact match / faithfulness checks)
- Export the KG to a `.ttl` file and load from disk

---


