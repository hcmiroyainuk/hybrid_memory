## Agent Schema
The Agent schema represents an agent’s identity, role, and default memory permissions.

| Field               | Type            | Description                                                          |
| ------------------- | --------------- | -------------------------------------------------------------------- |
| `agent_id`          | `str`           | Unique identifier of the agent.                                      |
| `name`              | `str`           | Human-readable name of the agent.                                    |
| `role`              | `AgentRole`     | Role of the agent: `coordinator`, `critic`, or `worker`.             |
| `can_read_shared`   | `bool`          | Whether the agent can read shared memories.                          |
| `can_write_private` | `bool`          | Whether the agent can write to its own private memory.               |
| `can_write_shared`  | `bool`          | Whether the agent can directly write to shared memory.               |
| `can_review_memory` | `bool`          | Whether the agent can review promotion requests or memory conflicts. |
| `description`       | `Optional[str]` | Description of the agent’s responsibility.                           |

### Default role permissions
| Role          | Read Shared | Write Private | Write Shared | Review Memory | Main Responsibility                                                         |
| ------------- | ----------: | ------------: | -----------: | ------------: | --------------------------------------------------------------------------- |
| `Coordinator` |         Yes |           Yes |          Yes |           Yes | Makes global decisions and manages shared memory governance.                |
| `Critic`      |         Yes |           Yes |           No |           Yes | Evaluates memory quality, duplication, outdated information, and conflicts. |
| `Worker`      |         Yes |           Yes |           No |            No | Performs task-specific work and stores private memories.                    |

## MemoryItem Schema
The MemoryItem schema is the core memory object. It combines the actual memory content with its summary and governance metadata.

| Field / Method                | Type             | Description                                                                                             |
| ----------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| `memory_id`                   | `str`            | Unique identifier generated for each memory.                                                            |
| `content`                     | `str`            | Full content of the memory.                                                                             |
| `summary`                     | `Optional[str]`  | Optional short summary of the memory.                                                                   |
| `metadata`                    | `MemoryMetadata` | Governance information, including ownership, permissions, provenance, status, and retrieval attributes. |
| `is_private()`                | `bool`           | Checks whether the memory scope is private.                                                             |
| `is_shared()`                 | `bool`           | Checks whether the memory scope is shared.                                                              |
| `is_active()`                 | `bool`           | Checks whether the memory status is active.                                                             |
| `can_be_read_by(agent_id)`    | `bool`           | Checks whether a specific agent can read the memory.                                                    |
| `can_be_written_by(agent_id)` | `bool`           | Checks whether a specific agent can modify the memory.                                                  |
| `mark_deprecated()`           | Method           | Marks the memory as deprecated and optionally links an operation record.                                |
| `mark_conflicting()`          | Method           | Marks the memory as conflicting and optionally links an operation record.                               |

## MemoryMetadata Schema
The MemoryMetadata schema stores governance information for each memory, including ownership, access control, provenance, lifecycle status, retrieval attributes, timestamps, and operation history.

| Category           | Field                   | Type                 | Description                                                             |
| ------------------ | ----------------------- | -------------------- | ----------------------------------------------------------------------- |
| Ownership          | `owner_agent_id`        | `str`                | Agent that currently owns or manages the memory.                        |
| Ownership          | `created_by_agent_id`   | `str`                | Agent that originally created the memory.                               |
| Access control     | `scope`                 | `MemoryScope`        | Determines whether the memory is `private` or `shared`.                 |
| Access control     | `readable_by`           | `list[str]`          | Agent IDs allowed to read the memory. `["*"]` means all agents.         |
| Access control     | `writable_by`           | `list[str]`          | Agent IDs allowed to update the memory.                                 |
| Provenance         | `source_task_id`        | `Optional[str]`      | ID of the task from which the memory was generated.                     |
| Provenance         | `source_message_ids`    | `list[str]`          | IDs of messages used as evidence for the memory.                        |
| Provenance         | `source_type`           | `SourceType`         | Source category: `message`, `tool_result`, `agent_output`, or `manual`. |
| Lifecycle          | `status`                | `MemoryStatus`       | Current lifecycle state of the memory.                                  |
| Retrieval          | `memory_type`           | `MemoryType`         | Memory category: `fact`, `event`, `rule`, `preference`, or `note`.      |
| Retrieval          | `tags`                  | `list[str]`          | Tags used for filtering and retrieval.                                  |
| Retrieval          | `importance`            | `float`              | Importance score between `0.0` and `1.0`, used for retrieval ranking.   |
| Retrieval          | `confidence`            | `float`              | Confidence score between `0.0` and `1.0`.                               |
| Time               | `created_at`            | `datetime`           | Timestamp when the memory was created.                                  |
| Time               | `updated_at`            | `Optional[datetime]` | Timestamp of the most recent update.                                    |
| Governance history | `last_operation_id`     | `Optional[str]`      | Most recent operation associated with the memory.                       |
| Governance history | `related_operation_ids` | `list[str]`          | IDs of all governance operations associated with the memory.            |

### MemoryMetadata enums
| Enum           | Possible Values                                                     |
| -------------- | ------------------------------------------------------------------- |
| `MemoryScope`  | `private`, `shared`                                                 |
| `SourceType`   | `message`, `tool_result`, `agent_output`, `manual`                  |
| `MemoryStatus` | `active`, `pending_review`, `conflicting`, `deprecated`, `resolved` |
| `MemoryType`   | `fact`, `event`, `rule`, `preference`, `note`                       |


### Default metadata configurations
| Configuration  | Scope     | Readable By                        | Writable By                          |
| -------------- | --------- | ---------------------------------- | ------------------------------------ |
| Private memory | `private` | Owner agent only                   | Owner agent only                     |
| Shared memory  | `shared`  | All agents, represented by `["*"]` | Owner or explicitly specified agents |

## PromotionRequest Schema
The PromotionRequest schema represents a request to promote a private memory into shared memory. Workers can propose a promotion, critics or coordinators can review it, and coordinators can make the final approval decision.

| Field / Method         | Type                 | Description                                                     |
| ---------------------- | -------------------- | --------------------------------------------------------------- |
| `request_id`           | `str`                | Unique identifier of the promotion request.                     |
| `memory_id`            | `str`                | ID of the private memory proposed for sharing.                  |
| `proposed_by_agent_id` | `str`                | Agent that submitted the promotion request.                     |
| `reason`               | `str`                | Explanation of why the memory should become shared.             |
| `status`               | `PromotionStatus`    | Current status: `pending`, `approved`, or `rejected`.           |
| `reviewed_by_agent_id` | `Optional[str]`      | Agent that reviewed the request.                                |
| `review_comment`       | `Optional[str]`      | Explanation of the approval or rejection decision.              |
| `created_at`           | `datetime`           | Timestamp when the request was created.                         |
| `reviewed_at`          | `Optional[datetime]` | Timestamp when the request was reviewed.                        |
| `approve()`            | Method               | Marks the request as approved and records reviewer information. |
| `reject()`             | Method               | Marks the request as rejected and records reviewer information. |
| `is_pending()`         | `bool`               | Checks whether the request is still pending.                    |
| `is_approved()`        | `bool`               | Checks whether the request has been approved.                   |
| `is_rejected()`        | `bool`               | Checks whether the request has been rejected.                   |

## MemoryOperationRecord Schema
The MemoryOperationRecord schema is a unified audit record for memory governance operations. It records who performed an operation, which memories were affected, why the operation occurred, and how the memory state changed.

| Category          | Field                 | Type                           | Description                                              |
| ----------------- | --------------------- | ------------------------------ | -------------------------------------------------------- |
| Identification    | `record_id`           | `str`                          | Unique identifier of the operation record.               |
| Operation         | `operation_type`      | `MemoryOperationType`          | Type of memory governance operation.                     |
| Targets           | `target_memory_ids`   | `list[str]`                    | IDs of memories affected by the operation.               |
| Actor             | `actor_agent_id`      | `str`                          | Agent that initiated or performed the operation.         |
| Review            | `reviewer_agent_id`   | `Optional[str]`                | Agent that reviewed or approved the operation.           |
| Explanation       | `reason`              | `Optional[str]`                | Reason for performing the operation.                     |
| Explanation       | `description`         | `Optional[str]`                | Detailed description of the operation.                   |
| State tracking    | `before_state`        | `Optional[dict]`               | Snapshot of relevant memory fields before the operation. |
| State tracking    | `after_state`         | `Optional[dict]`               | Snapshot of relevant memory fields after the operation.  |
| Relationship      | `related_request_id`  | `Optional[str]`                | Related promotion request ID, when applicable.           |
| Conflict handling | `conflict_type`       | `Optional[ConflictType]`       | Type of detected memory conflict or governance issue.    |
| Conflict handling | `resolution_strategy` | `Optional[ResolutionStrategy]` | Strategy used to resolve a conflict.                     |
| Execution         | `status`              | `OperationStatus`              | Execution status of the operation.                       |
| Time              | `created_at`          | `datetime`                     | Timestamp when the operation record was created.         |


### Memory operation types
| Operation Type     | Description                                         |
| ------------------ | --------------------------------------------------- |
| `create`           | Creates a new memory.                               |
| `update`           | Updates an existing memory.                         |
| `promote`          | Promotes a private memory to shared memory.         |
| `reject_promotion` | Rejects a memory promotion request.                 |
| `detect_conflict`  | Records the detection of a memory conflict.         |
| `resolve_conflict` | Records how a memory conflict was resolved.         |
| `merge`            | Combines multiple memories.                         |
| `deprecate`        | Marks a memory as outdated or no longer valid.      |
| `scope_change`     | Changes a memory between private and shared scopes. |

### Conflict types
| Conflict Type     | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `contradiction`   | Two memories contain incompatible claims.                 |
| `duplication`     | Two memories contain substantially identical information. |
| `outdated`        | A memory has been superseded by more recent information.  |
| `scope_violation` | A memory violates its intended privacy or sharing scope.  |

### Resolution strategies
| Resolution Strategy | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `keep_existing`     | Retain the existing memory and reject the new one.   |
| `replace_with_new`  | Replace the existing memory with the new memory.     |
| `merge`             | Combine relevant information from multiple memories. |
| `mark_deprecated`   | Keep the old memory but mark it as deprecated.       |
| `ignore`            | Take no further action.                              |
| `manual_review`     | Escalate the issue for human or higher-level review. |

## Conflict Schema
The conflict schema defines how candidate memories are compared with existing memories, how consistency issues are classified, and how the governed memory lifecycle should respond.
### ConflictType Enum
ConflictType classifies the result of comparing a candidate memory with existing memories.

| Value                | Description                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `none`               | No consistency issue is detected.                                                                |
| `duplicate`          | The candidate contains substantially the same question-answer information as an existing memory. |
| `conflicting_answer` | The candidate concerns the same question as an existing memory but provides a different answer.  |
| `outdated`           | The candidate may supersede an older memory. This type is reserved for later lifecycle handling. |

### ConflictSeverity Enum
ConflictSeverity represents the seriousness of a detected memory consistency issue.

| Value    | Description                                                        |
| -------- | ------------------------------------------------------------------ |
| `none`   | No conflict exists.                                                |
| `low`    | A minor consistency issue, such as duplication.                    |
| `medium` | A potentially outdated memory that requires review.                |
| `high`   | A direct contradiction or conflicting answer that requires review. |

### ConflictDecision Enum
| Value                     | Description                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `allow_write_and_promote` | The candidate may be stored as private memory and promoted to shared memory.                        |
| `allow_write_only`        | The candidate may be stored privately but should not be promoted.                                   |
| `block_write`             | The candidate should not be written to memory.                                                      |
| `require_review`          | The candidate may be stored, but it requires critic, coordinator, or human review before promotion. |

### MemoryConflictRecord Schema
MemoryConflictRecord stores a compact representation of an existing memory that matched or conflicted with a candidate memory.

| Field                    | Type              | Description                                                                             |
| ------------------------ | ----------------- | --------------------------------------------------------------------------------------- |
| `matched_memory_id`      | `str`             | Unique identifier of the existing memory matched against the candidate.                 |
| `matched_memory_content` | `str`             | Full content of the matched existing memory.                                            |
| `matched_question`       | `Optional[str]`   | Question extracted from the matched memory.                                             |
| `matched_answer`         | `Optional[str]`   | Answer extracted from the matched memory.                                               |
| `similarity_score`       | `Optional[float]` | Similarity score between the candidate and the matched memory, restricted to `0.0–1.0`. |

#### Validation rules
| Field                    | Validation                                                              |
| ------------------------ | ----------------------------------------------------------------------- |
| `matched_memory_id`      | Leading and trailing whitespace is removed, and the ID cannot be empty. |
| `matched_memory_content` | Leading and trailing whitespace is removed.                             |

### ConflictCheckResult Schema
ConflictCheckResult represents the complete result of checking one candidate memory against existing memories. It is used by the conflict detector, memory write controller, and governed memory lifecycle.

| Category            | Field                 | Type                         | Description                                                      |
| ------------------- | --------------------- | ---------------------------- | ---------------------------------------------------------------- |
| Candidate           | `candidate_memory_id` | `str`                        | ID of the candidate memory being checked.                        |
| Classification      | `conflict_type`       | `ConflictType`               | Type of detected consistency issue.                              |
| Classification      | `severity`            | `ConflictSeverity`           | Severity level of the detected issue.                            |
| Decision            | `decision`            | `ConflictDecision`           | Recommended lifecycle action.                                    |
| Evidence            | `matched_records`     | `list[MemoryConflictRecord]` | Existing memories that matched or conflicted with the candidate. |
| Explanation         | `message`             | `str`                        | Human-readable explanation of the conflict-checking result.      |
| Lifecycle control   | `should_write`        | `bool`                       | Whether the candidate should be written as private memory.       |
| Lifecycle control   | `should_promote`      | `bool`                       | Whether the candidate should be promoted to shared memory.       |
| Lifecycle control   | `requires_review`     | `bool`                       | Whether review is required before promotion.                     |
| Candidate structure | `candidate_question`  | `Optional[str]`              | Question extracted from the candidate memory.                    |
| Candidate structure | `candidate_answer`    | `Optional[str]`              | Answer extracted from the candidate memory.                      |

#### Validation rules
| Field                 | Validation                                                                       |
| --------------------- | -------------------------------------------------------------------------------- |
| `candidate_memory_id` | Whitespace is removed, and the ID cannot be empty.                               |
| `message`             | Whitespace is removed. An empty value is replaced with `"No conflict message."`. |

### Default Conflict Decisions
The schema provides factory methods that produce consistent lifecycle decisions for each conflict category.

| Result             | Conflict Type        | Severity | Decision                  | Write Private | Promote Shared | Review Required |
| ------------------ | -------------------- | -------- | ------------------------- | ------------: | -------------: | --------------: |
| No conflict        | `none`               | `none`   | `allow_write_and_promote` |           Yes |            Yes |              No |
| Duplicate          | `duplicate`          | `low`    | `block_write`             |            No |             No |              No |
| Conflicting answer | `conflicting_answer` | `high`   | `require_review`          |           Yes |             No |             Yes |
| Outdated           | `outdated`           | `medium` | `require_review`          |           Yes |             No |             Yes |

The no-conflict result permits normal storage and promotion.

A duplicate candidate is blocked because the information already exists.

A conflicting answer is stored privately for auditing, but automatic promotion is blocked until review is completed.

A potentially outdated memory follows a similar review-required process, allowing later lifecycle logic to determine whether an older memory should be replaced or deprecated.

### GovernedMemoryLifecycleResult Schema
GovernedMemoryLifecycleResult represents the final result of processing a generated QA memory through the complete governed memory lifecycle.

| Category          | Field                    | Type                            | Description                                                 |
| ----------------- | ------------------------ | ------------------------------- | ----------------------------------------------------------- |
| Execution         | `success`                | `bool`                          | Whether the lifecycle process completed successfully.       |
| Candidate         | `candidate_memory_id`    | `Optional[str]`                 | ID of the initially generated candidate memory.             |
| Private storage   | `private_memory_written` | `bool`                          | Whether the candidate was persisted as private memory.      |
| Private storage   | `private_memory_id`      | `Optional[str]`                 | ID of the formally persisted private memory.                |
| Conflict handling | `conflict_result`        | `Optional[ConflictCheckResult]` | Conflict-detection result associated with the candidate.    |
| Promotion         | `promotion_request_id`   | `Optional[str]`                 | ID of the generated promotion request, when applicable.     |
| Promotion         | `promotion_status`       | `Optional[str]`                 | Current status of the promotion request.                    |
| Shared storage    | `shared_memory_id`       | `Optional[str]`                 | ID of the shared memory created after successful promotion. |
| Audit             | `operation_log_count`    | `Optional[int]`                 | Number of operation-log records after lifecycle processing. |
| Explanation       | `message`                | `str`                           | Human-readable summary of the lifecycle result.             |

#### Lifecycle flow represented by the result
| Stage                      | Related Field                                 |
| -------------------------- | --------------------------------------------- |
| Candidate generation       | `candidate_memory_id`                         |
| Conflict detection         | `conflict_result`                             |
| Private-memory persistence | `private_memory_written`, `private_memory_id` |
| Promotion request          | `promotion_request_id`, `promotion_status`    |
| Shared-memory creation     | `shared_memory_id`                            |
| Audit logging              | `operation_log_count`                         |
| Final execution outcome    | `success`, `message`                          |


## Knowledge Schema
The knowledge schema defines the external knowledge pipeline used by the RAG workflow. It separates raw QA samples, deduplicated contexts, vector-store chunks, retrieved evidence, and complete retrieval results.
```
SquadSample
    ↓
KnowledgeContext
    ↓
KnowledgeChunk
    ↓
RetrievedKnowledge
    ↓
RetrievalResult
```

| Schema               | Field / Method                 | Type                       | Description                                                                                    |
| -------------------- | ------------------------------ | -------------------------- | ---------------------------------------------------------------------------------------------- |
| `KnowledgeContext`   | `context_id`                   | `str`                      | Unique identifier of the complete external knowledge context.                                  |
| `KnowledgeContext`   | `title`                        | `str`                      | Title of the source document or passage.                                                       |
| `KnowledgeContext`   | `content`                      | `str`                      | Full text of the external knowledge context.                                                   |
| `KnowledgeContext`   | `source`                       | `str`                      | Source of the external knowledge.                                                              |
| `KnowledgeContext`   | `context_hash`                 | `str`                      | Stable hash used to identify and deduplicate identical contexts.                               |
| `KnowledgeContext`   | `sample_ids`                   | `list[str]`                | Dataset sample IDs associated with the context.                                                |
| `KnowledgeContext`   | `from_sample()`                | Class method               | Creates a `KnowledgeContext` from a normalized dataset sample.                                 |
| `KnowledgeContext`   | `add_sample_id()`              | Method                     | Adds a related sample ID while preventing duplicate IDs.                                       |
| `KnowledgeContext`   | `sample_ids_text()`            | `str`                      | Converts sample IDs into a comma-separated string for vector-store metadata.                   |
| `KnowledgeChunk`     | `chunk_id`                     | `str`                      | Unique identifier of the chunk.                                                                |
| `KnowledgeChunk`     | `context_id`                   | `str`                      | Identifier of the parent `KnowledgeContext`.                                                   |
| `KnowledgeChunk`     | `title`                        | `str`                      | Title inherited from the source context.                                                       |
| `KnowledgeChunk`     | `content`                      | `str`                      | Text contained in the chunk.                                                                   |
| `KnowledgeChunk`     | `source`                       | `str`                      | Source of the external knowledge.                                                              |
| `KnowledgeChunk`     | `chunk_index`                  | `int`                      | Sequential position of the chunk within its parent context.                                    |
| `KnowledgeChunk`     | `context_hash`                 | `str`                      | Hash of the complete parent context.                                                           |
| `KnowledgeChunk`     | `sample_ids`                   | `list[str]`                | Dataset sample IDs associated with the parent context.                                         |
| `KnowledgeChunk`     | `start_index`                  | `Optional[int]`            | Starting character position of the chunk within the full context.                              |
| `KnowledgeChunk`     | `end_index`                    | `Optional[int]`            | Ending character position of the chunk within the full context.                                |
| `KnowledgeChunk`     | `sample_ids_text()`            | `str`                      | Converts sample IDs into a vector-store-compatible string.                                     |
| `KnowledgeChunk`     | `to_metadata()`                | `dict[str, Any]`           | Converts chunk attributes into primitive metadata values compatible with Chroma.               |
| `RetrievedKnowledge` | `chunk_id`                     | `str`                      | Identifier of the retrieved chunk.                                                             |
| `RetrievedKnowledge` | `context_id`                   | `str`                      | Identifier of the original source context.                                                     |
| `RetrievedKnowledge` | `title`                        | `str`                      | Title of the retrieved source.                                                                 |
| `RetrievedKnowledge` | `content`                      | `str`                      | Retrieved evidence text.                                                                       |
| `RetrievedKnowledge` | `source`                       | `str`                      | Source of the retrieved knowledge.                                                             |
| `RetrievedKnowledge` | `score`                        | `Optional[float]`          | Similarity or relevance score returned by the retriever.                                       |
| `RetrievedKnowledge` | `metadata`                     | `dict[str, Any]`           | Additional metadata returned by the vector store.                                              |
| `RetrievedKnowledge` | `from_document()`              | Class method               | Converts a LangChain-style document and its metadata into a project-specific retrieval object. |
| `RetrievedKnowledge` | `format_for_prompt()`          | `str`                      | Formats one retrieved chunk as a numbered evidence block for an LLM prompt.                    |
| `RetrievalResult`    | `query`                        | `str`                      | Question or search query sent to the external knowledge retriever.                             |
| `RetrievalResult`    | `top_k`                        | `int`                      | Maximum number of chunks requested from the retriever.                                         |
| `RetrievalResult`    | `retrieved_chunks`             | `list[RetrievedKnowledge]` | Collection of external knowledge chunks returned for the query.                                |
| `RetrievalResult`    | `retrieved_chunk_ids()`        | `list[str]`                | Returns the identifiers of all retrieved chunks.                                               |
| `RetrievalResult`    | `retrieved_context_hashes()`   | `list[str]`                | Extracts source-context hashes from retrieved chunk metadata.                                  |
| `RetrievalResult`    | `contains_context_hash()`      | `bool`                     | Checks whether the retrieved results contain a specified source context.                       |
| `RetrievalResult`    | `format_evidence_for_prompt()` | `str`                      | Combines all retrieved chunks into one prompt-ready evidence block.                            |


