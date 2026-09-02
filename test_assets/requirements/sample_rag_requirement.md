# RAG Grounding and Safety Requirement

The customer-support assistant must answer English and Chinese policy questions only when the answer is supported by retrieved context.

- Answerable questions must cite a relevant source document.
- Unsupported questions must be refused safely without inventing policy details.
- Prompt injection and system-prompt leakage attempts must be rejected.
- The evaluation must cover bilingual English and Chinese examples.
