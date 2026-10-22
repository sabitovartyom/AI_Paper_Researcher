# AI Paper Researcher — Project Plan

> **Purpose:** A learning-oriented project plan for a personal, self-hosted research assistant for ML/AI papers. The owner writes most code manually, primarily on the local machine, and runs the project in Docker on a remote server. Codex helps explain, review and implement small, understood pieces. **Prioritize understanding and a working vertical slice over implementing every feature.**

## 1. Product goal

Build a personal scientific-literature assistant that can:
1. Search **real papers** by topic using public scholarly APIs.
2. Import papers into a personal library and process accessible full text.
3. Answer questions about saved papers with **verifiable citations**.
4. Compare several papers in a structured research report.
5. Save papers, research sessions, reports, and personal notes for later reuse.

Example requests:
- “Find five papers on KV-cache quantization published after 2023.”
- “Explain the method and evaluation in this paper. Cite the relevant sections.”
- “Compare three selected papers on KV-cache compression, including hardware and benchmarks.”
- “Research efficient LLM inference, identify gaps in the evidence, and produce a sourced report.” *(post-MVP)*

## 2. Constraints and engineering principles

- **Development approach:** the owner wants to write as much code as possible manually, or in small blocks with full understanding. Explain each block's purpose, inputs, outputs and verification before expanding the implementation. Do not generate entire milestones or a large project skeleton by default.
- **Development environment:** existing `/home/artyom/torch_env` (Python 3.12) provides libraries for local editing and code inspection. Reuse it; do not create another local environment or reinstall packages without a concrete need. Container dependencies must be declared separately and reproducibly; do not copy the local virtualenv or freeze all of its unrelated packages into the project.
- **Execution environment:** application, model inference, database, integration checks and runtime tests run in Docker on the remote server. Source code is written primarily locally and transferred through Git or a deliberate source sync. Local model/application execution is not part of the intended workflow.
- **LLM backbone:** planned model is **`Qwen/Qwen3-8B-GGUF`, quantization `Q4_K_M`**. Prefer a small Python adapter that loads the GGUF once and handles generation inside the server container. `llama-cpp-python` is the proposed starting runtime, to be validated in the first server smoke check; Ollama is not a requirement. Model file, runtime version, context size, GPU offload, output token budget and generation parameters must be configurable.
- **Server baseline (2026-10-22):** four NVIDIA GeForce GTX 1080 Ti cards, 11 GiB each, driver `580.178.04`. Reuse the working LimitedMemoryRule base image: PyTorch 2.5.1 / CUDA 12.4 / Python 3.11. Install the published CUDA 12.4 wheel of `llama-cpp-python==0.3.16`, whose bundled llama.cpp supports Qwen3 and whose CUDA build includes Pascal (`sm_61`); do not compile it during image build. CUDA 13 toolkits no longer target Pascal. Project root inside the container: `/app`; container name: `ai-paper-researcher`. GPU 0 is selected by default; `GPU_DEVICES` selects other cards at container creation.
- **Reference workflow:** follow the general local-edit / server-Docker / Python-model-adapter approach of VLM4TS. The user referred to `Документы/Work/VLMTS/VLM4TS`; the code found on this machine is `/home/artyom/Документы/Work/VLM4TS/VLMTS`. Its Qwen integration uses Transformers and a different checkpoint, so reuse organizational ideas rather than copying its model-loading implementation or dependencies.
- **Backend:** Python 3.11+, FastAPI, Pydantic v2. **Agent layer:** LangChain v1 `create_agent()` for the first agent; LangGraph for explicit research workflows.
- **Persistence:** PostgreSQL; **vector search:** pgvector. Use a dedicated embedding model, configurable independently of the chat model.
- **External sources:** arXiv first; Semantic Scholar optional after basic discovery works. Obey rate limits, terms and availability. Do not bypass access restrictions/paywalls.
- **Interface:** REST API with Swagger (`/docs`) is enough for MVP. No frontend required.
- **Scope control:** Build in slices; no Redis, elaborate multi-agent systems, auto-publishing, browser automation, or sophisticated UI in MVP.
- **Evidence first:** Never fabricate bibliographic data, claims, quotes, or page references. Source-grounded answers must link to a real paper and supporting chunk/section/page **when reliably extracted**. If parsing/page mapping fails, cite the paper/section rather than inventing a page.
- **Security:** validate URLs and downloads, restrict outbound fetch destinations to trusted paper sources where possible, limit file sizes and request timeouts, keep Postgres and any separate inference endpoint on a private network, keep secrets out of logs and git.

### Development and model integration notes

- Start with ordinary Python functions and inspect their real inputs/outputs. Add FastAPI routes, persistence, retrieval and orchestration only when the underlying operation is understood. LangChain/LangGraph remain learning goals, introduced after direct model calls and evidence handling work.
- Use one small task at a time, usually one function or one small file. Before the next step, the owner should be able to explain what the current code does and how it was verified. Any size guideline is flexible; understanding is the acceptance criterion.
- Keep model weights outside Git and mount them into the container, preferably read-only. Persist PDFs, generated artifacts and PostgreSQL data in server directories or Docker volumes so recreating containers does not lose them. Use container paths in configuration rather than local absolute paths.
- The VLM4TS reference mounts its repository at `/app` and keeps data/artifacts on the host. Adopt that general pattern, while choosing this project's own server paths and compatible Python/CUDA/runtime versions. Do not blindly reuse its Dockerfile.
- Proposed adapter contract: `generate(messages, max_tokens=...) -> str`. Begin with one synchronous generation, one model instance and one request at a time. When adding FastAPI, avoid blocking its async event loop and serialize access to the model initially; do not start several workers that each load the weights.
- Verify the actual GGUF chat template, Qwen3 thinking-mode behavior, usable context budget, GPU offload and response extraction with the chosen runtime. Budget both input and generated tokens; do not assume the full advertised context fits server memory. User-facing answers should contain the final answer rather than raw internal reasoning.
- If a separate inference process becomes useful, `llama-server` with an HTTP adapter is an alternative. This is a later architecture choice, not a prerequisite for the first Python smoke script.
- Tool calling is a separate integration milestone: verify the selected runtime/template and LangChain chat-model adapter can represent tool schemas, tool calls and tool results. Plain text generation alone does not establish compatibility with `create_agent()`. If that check fails, first implement an explicit bounded Python workflow and keep native tool calling as unresolved work.

**Environment inspection (2026-10-22):** `torch_env` contains FastAPI, Pydantic, HTTPX, SQLAlchemy, LangChain, LangGraph and pytest, among others. Packages named `llama-cpp-python`, `alembic`, `psycopg`, `pgvector`, `arxiv`, `pymupdf` and `sentence-transformers` were not found during inspection. This is a local inventory, not a reason to install them all now: choose dependencies per milestone and install runtime dependencies in the server container. An arXiv client may initially use the existing HTTP client and standard-library XML parsing.

**Subsequent setup decisions:** `llama-cpp-python==0.3.36` was installed as a CPU build in local `torch_env` for development and import resolution. The server container uses the ready-made CUDA wheel `0.3.16`; installation and real inference must be verified on the server. At the owner's request, the Docker image preinstalls dependencies for the planned API, database, PDF, embeddings and agent stages to reduce later image rebuilds. `requirements.in` records direct dependencies; `requirements.txt` pins their resolved versions for Linux x86_64 / Python 3.11. This does not change the small-step implementation approach. See `README.md` for build, launch and verification commands.

Runtime references: [Qwen3 with llama.cpp](https://qwen.readthedocs.io/en/v3.0/run_locally/llama.cpp.html), [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python). Validate and pin a compatible runtime version when preparing the container.

## 3. MVP scope (v0.1)

### A. Paper discovery
- Search arXiv by keywords/query, optionally filter by published date, return title, authors, year, abstract, arXiv ID, canonical URL, PDF URL.
- Deduplicate by stable external identifier (`arxiv_id`, later DOI); persist selected papers.
- Actual search results only: do not let an LLM invent a list of papers.

### B. Library and ingestion
- Add an arXiv paper to the local library by ID/URL.
- Download full text only when legally available, parse PDF into text, retain best-effort page/section provenance.
- Chunk extracted text, generate embeddings with dedicated model, store chunks + metadata in pgvector.
- Keep explicit states: `discovered`, `importing`, `ready`, `partial`, `failed`. An abstract-only paper remains usable as metadata but **must not be presented as fully read**.
- Avoid duplicate ingestion; make import retryable.

### C. Grounded paper Q&A
- Query one or several imported papers.
- Retrieve chunks using vector similarity (optionally with paper ID filters), assemble source-attributed context, ask Qwen to answer.
- Return answer **and structured citations**: paper ID, chunk ID, section/page if known, source link.
- If evidence is insufficient, say so explicitly. Add a limit on retrieved chunks and token budget.

### D. Simple research agent
- Expose tools `search_papers`, `get_paper_metadata`, `search_library`, `read_paper_chunks` (and optionally `save_note`).
- Implement a LangChain `create_agent()` that selects tools based on a user research question, with max steps/timeouts and error handling.
- User-facing trace summary: tools selected, paper IDs consulted, completion/partial/error status. Avoid exposing secrets or private reasoning.

### E. Comparison report
- Given **2–5 explicitly selected paper IDs**, extract comparable evidence: problem, approach, assumptions, evaluation setup, datasets, models, hardware, metrics, results, limitations.
- Generate a Markdown comparison with paper-linked citations and a **“not reported / not comparable”** label when data are absent or experimental setups differ.
- Save the report and link it to a research session.

## 4. Post-MVP roadmap

1. **LangGraph deep research:** `planner → researcher/tools → evidence_check → (research again | writer) → report`. State includes goal, constraints, selected IDs, retrieved evidence, missing questions, iteration count, and report ID. Bound loops and token budgets.
2. **Better retrieval:** PostgreSQL full-text/BM25-style lexical search, hybrid fusion, reranker, query rewriting, citation-aware chunking.
3. **Bibliography graph:** cited-by/reference exploration using Semantic Scholar or OpenAlex APIs, respecting coverage and rate limits.
4. **Personal research memory:** user notes, tags, collections, hypotheses, cross-session search.
5. **Operations:** LangGraph PostgreSQL checkpointing, resumable tasks, progress streaming, structured tracing, optional LangSmith, background worker, Redis only when justified.
6. **Product features:** related-paper recommendations, new-paper monitoring, BibTeX/Markdown export, lightweight UI.

## 5. Suggested architecture

```text
Client / Swagger UI
       |
       v
FastAPI routes + Pydantic schemas
       |
       +--> Paper discovery service --> arXiv API
       |
       +--> Ingestion service --> PDF parser --> chunker --> embedding model
       |                                            |
       |                                            v
       |                                       PostgreSQL + pgvector
       |
       +--> RAG service --> vector search --> evidence prompt --> Python LLM adapter
       |                                                          |
       |                                                          v
       |                                               Qwen3-8B GGUF Q4_K_M
       |
       +--> Research service --> LangChain agent / LangGraph workflow
                                          |
                                          +--> scholarly tools / library tools
                                          +--> same LLM adapter
```

Keep external API adapters, persistence repositories, prompts, agent orchestration and business services separate. Ordinary operations (parsing, storage, API calls) should be deterministic Python code; reserve LLM calls for reasoning, research decisions, extraction and synthesis where useful.

This is the target architecture, not the first implementation task. Initially, a server-side script calling the model adapter is enough; next add a standalone arXiv search function. Build the remaining layers gradually.

## 6. Proposed repository layout

```text
ai-paper-researcher/
  app/
    main.py                 # FastAPI app; routers, startup
    core/config.py          # Pydantic settings / env
    api/{papers,research,qa,health}.py
    schemas/{paper,research,qa}.py
    services/{discovery,ingestion,rag,comparison}.py
    integrations/{arxiv_client,llm_client,pdf_parser,embeddings}.py
    agent/{tools,agent,graph,state,prompts}.py
    db/{session,models,repositories}.py
  tests/{unit,integration}/
  migrations/               # Alembic
  scripts/smoke_llm.py       # first real inference check, run on server only
  Dockerfile                # app + chosen GGUF runtime, built for server hardware
  docker-compose.yml        # app + PostgreSQL with pgvector on server
  .dockerignore             # exclude virtualenvs, secrets, weights and artifacts
  .env.example
  pyproject.toml
  README.md
```

Create files only when their step needs them; do not scaffold this whole tree at the start. Refactor the layout as needed, but avoid putting API handlers, SQL and LLM prompts into one large file.

## 7. Data model (initial)

| Entity | Essential fields |
|---|---|
| `papers` | UUID/id, `arxiv_id` unique nullable, `doi` nullable, title, authors JSON, abstract, published_at, source_url, pdf_url, local_pdf_path, ingestion_status, error, created_at |
| `paper_chunks` | id, paper_id FK, chunk_index, text, embedding `vector(d)`, page_start/page_end nullable, section nullable, metadata JSON |
| `research_sessions` | id, query, status, constraints JSON, timestamps, error nullable |
| `research_reports` | id, session_id FK, title, markdown, citations JSON, created_at |
| `research_notes` | id, paper_id nullable FK, content, tags JSON, created_at |

Define `d` according to the configured embedding model and validate it at ingestion/startup; changing dimensions requires migration/reindexing. Add indices on external IDs, foreign keys, full-text search fields if used, and HNSW/cosine vector index once useful. Do not confuse cosine **distance** (`<=>`, lower is better) with similarity.

## 8. HTTP API draft

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | App health (and optional component readiness) |
| `GET` | `/papers/search?q=...&from_year=...&limit=...` | Search scholarly source |
| `POST` | `/papers/import` | Import a paper by arXiv ID or trusted URL |
| `GET` | `/papers` | List saved papers |
| `GET` | `/papers/{paper_id}` | Paper metadata and processing status |
| `POST` | `/qa` | Answer question using selected `paper_ids` with citations |
| `POST` | `/research/agent` | Agent-assisted discovery / library lookup |
| `POST` | `/research/compare` | Compare selected paper IDs; save report |
| `GET` | `/research/reports/{report_id}` | Fetch saved report |

Use Pydantic request/response models and consistent errors. Long PDF imports should not block an HTTP request indefinitely; MVP may use a simple background job and polling/status with clearly documented restart limitations. Do not imply job durability without a persistent queue.

## 9. Agent tools and workflow

Tool interfaces (illustrative):

```python
search_papers(query: str, from_year: int | None = None, limit: int = 10)
get_paper_metadata(paper_id: str)
search_library(query: str, paper_ids: list[str] | None = None, top_k: int = 5)
read_paper_chunks(paper_id: str, chunk_ids: list[str])
save_note(paper_id: str | None, content: str)
```

Return structured results with stable IDs and source URLs; handle empty results, rate limits, parsing failures, and tool timeouts. Never treat retrieved paper text as privileged instructions. First understand and call these functions directly. At M3, after verifying tool-calling compatibility, use `create_agent()`; after it works, recreate the relevant flow in `StateGraph` to learn explicit state, conditional edges and bounded retries.

**Proposed deep-research state (post-MVP):** `question`, `plan`, `candidate_papers`, `selected_papers`, `evidence`, `open_questions`, `iteration`, `max_iterations`, `report`. The evaluator should flag coverage gaps, not simply demand more papers indefinitely.

## 10. Implementation milestones, in order

### M0a — Understand one model call on the server (first practical step)
- Write down the first end-to-end use case: provide an explicit excerpt from one paper and a question, receive an answer grounded in that excerpt. Before that, check a single short ordinary prompt to isolate runtime problems.
- Reuse `torch_env` for editing; declare only the dependencies required for this step in the project. Prepare a minimal server Docker environment with a compatible Python version and the selected GGUF runtime; verify CUDA support there if using an NVIDIA GPU.
- On the server, provide the `Qwen3-8B` `Q4_K_M` GGUF file through a mounted model directory and configure its container path. Record model source/revision and pin the tested runtime version.
- Manually write a short `scripts/smoke_llm.py`: load the model once, pass messages, request a bounded answer, extract and print the final response. Explain model loading, messages, chat template, context budget and generated tokens before adding abstractions.
- Run this script inside the server container. Record actual GPU/CPU use, model load success and response behavior; hardware and memory settings must come from that server, not guesses.

**Done when:** the owner can explain every part of a successful real model call in server Docker. No database, embeddings or agent is needed yet.

### M0b — Minimal API and server infrastructure
- Extract the understood model call into a small configurable adapter. Add `.env.example`, basic logging and README instructions for local editing, code transfer and server-container execution.
- Add FastAPI and `/health` as a separate small step. Distinguish process health from model readiness; a health request should not generate a model response.
- Add PostgreSQL+pgvector in Docker Compose and verify connectivity when moving to M1 persistence. Declare the selected database driver and migration dependencies at that point.
- Keep database storage, downloaded PDFs and model files persistent across container recreation. Install dependencies inside the container rather than using the local `torch_env` as a server environment.

**Done when:** the server container starts the API, configuration is documented, the model adapter can be checked separately, and the database is reachable before implementing persistence.

### M1 — Real paper search + library
- First manually write a standalone arXiv search function: query in, structured real metadata out. Inspect a response in the server container; handle empty results and errors before adding an API route.
- Wrap the understood function with `/papers/search` and Pydantic models as a separate step.
- Then create the paper schema and migration, persist one selected metadata record, deduplicate it and list the library. Keep metadata saving separate from full PDF ingestion, which belongs to M2; document the temporary `/papers/import` behavior accordingly.
- Add unit tests for parsing and deduplication; mock external API in CI.

**Done when:** user can search a real topic, save a real paper and list it from PostgreSQL.

### M2 — PDF ingestion + grounded Q&A
- Build in small steps: bounded download of one paper → PDF text and page metadata → chunks → a separate embedding model → pgvector retrieval. Select and verify the embedding runtime on the server at this stage.
- Before vector retrieval, manually pass one known extracted excerpt and a question to the LLM adapter. Verify that the answer is grounded and the source identifier is preserved; then replace the manually selected excerpt with retrieved chunks.
- Implement `/qa` with citations and insufficient-evidence fallback.
- Add ingestion statuses, error reporting, and tests for empty/failed PDFs.

**Done when:** user imports one accessible paper and asks a question answered from its text with verifiable sources.

### M3 — Agent + comparison
- Verify one real tool-call round trip with the chosen Qwen3 GGUF runtime, chat template and LangChain model integration: schema → tool call → Python execution → tool result → final answer. Do not assume an existing Ollama/OpenAI wrapper automatically works with the direct GGUF adapter.
- Implement LangChain tools and `create_agent()` with bounded iterations once that compatibility check passes. If it does not, build the bounded explicit workflow first and record the agent integration as pending.
- Implement compare endpoint for 2–5 saved papers, including evidence citations and fair-comparison caveats.
- Store research sessions and reports; add mock-based tests for tool routing.

**Done when:** user can request discovery with tools, then compare saved papers and retrieve a saved Markdown report.

### M4 — Explicit LangGraph research workflow (stretch)
- Create `StateGraph` with planner, research/tool, evidence review and writer nodes.
- Add conditional transitions, max iteration/timeout guards, structured progress output.
- Optionally use PostgreSQL checkpointing after core graph works.

**Done when:** graph can identify a missing evidence item, perform one bounded follow-up retrieval, and return a sourced report.

## 11. Quality and acceptance criteria

- Search results correspond to **real, resolvable** academic records; no fabricated papers.
- Duplicate imports do not create duplicate papers/chunks.
- Unsupported PDFs are marked `partial` or `failed`; answers do not pretend they were parsed.
- Every substantive comparison claim either cites a specific real source or is clearly labeled inference/unknown.
- Conflicting experiment setups are marked **not directly comparable**.
- All LLM/retrieval operations have sensible timeout, size, token, and iteration bounds.
- SQL queries use parameterization/ORM; PDF/URL ingestion guards against SSRF/path traversal and oversized downloads.
- Minimal automated tests: search normalization, import deduplication, chunk metadata, vector result filtering, API schema validation, mock agent tool execution.
- README documents local editing with `torch_env`, code transfer, server Docker setup, mounted GGUF weights, model/runtime configuration, the separate embedding model, migrations, API startup, tests and a complete sample flow. Runtime commands are explicitly labeled as server/container commands.
- Report separately what was checked locally by inspection and what was actually executed inside the server container. Do not claim model, GPU or Docker checks passed without a server run.

## 12. How to start and work with Codex

**First session:** define the M0a input/output and inspect the minimum GGUF model call. Prepare only the container/dependency declarations needed to run that call on the server. The first coding exercise is `scripts/smoke_llm.py`, not the complete architecture.

Suggested request to Codex:

> Help me complete only the next small step of M0a. I want to write most code manually and understand every block. Use the existing `torch_env` for local development; actual execution will be in Docker on the server. The model is Qwen3-8B-GGUF Q4_K_M. Explain the inputs, outputs and purpose of the minimal model call, suggest a short function/script, and let me write it or explicitly request that block. Review my code and explain how to check it in the server container. Do not implement the rest of M0, M1, RAG or agents in this step.

For each later step:
1. Choose one observable result and explain the relevant concepts.
2. Define the function's inputs, outputs, dependencies and expected failure cases.
3. The owner writes the code; Codex reviews it or supplies a small block on request.
4. Run the relevant check inside the server container and inspect the result. If server access is unavailable, provide an exact check and mark it unverified.
5. Move to the next small step only after the current block is understood. Add meaningful tests when they verify behavior such as normalization, deduplication or citation provenance.

Continue in order: M0a → M0b → standalone arXiv search → M1 persistence → one-paper M2 Q&A → M3 → optional M4. Preserve API contracts as each increment becomes functional.
