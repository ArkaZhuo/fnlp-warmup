# Research Report

- Query: How to improve RAG factuality?
- Generated At: 2026-04-10 20:00:00
- Report Format: Markdown only (HTML disabled by design)

## Executive Summary

This draft was manually edited by user.
Please add stronger comparison between retrieval strategies and include practical deployment guidance.

## Semantic Scholar Findings

1. **Paper A (edited by user)**
   - User notes: strengthen section and include deployment costs.

## Evidence From User Inputs

1. Source: `samples/sample_notes.md`
   - Snippet: User highlighted citation grounding.

## Deployment Checklist and Risk Controls

### Deployment Checklist
- [ ] **Retrieval Strategy Validation**: Compare dense vs. sparse vs. hybrid retrieval on target domain data for precision/recall.
- [ ] **Factuality Guardrails**: Implement pre-generation fact verification and post-generation citation accuracy checks.
- [ ] **Model & Infrastructure**: Select LLM (cost vs. accuracy), configure vector database, and establish latency/throughput SLAs.
- [ ] **Monitoring & Logging**: Deploy pipelines to track hallucination rates, retrieval hit rates, and user feedback.
- [ ] **Rollout Plan**: Begin with a limited beta, A/B test against baseline, and define success metrics.

### Risk Controls
- **Hallucination Mitigation**: Use constrained decoding, prompt engineering for uncertainty expression, and integrate a secondary verification model [1].
- **Data Poisoning & Bias**: Curate and audit the knowledge base; implement data provenance tracking and periodic retraining on updated corpora.
- **Performance Degradation**: Set up automated alerts for drops in retrieval quality or generation coherence; maintain a fallback response mechanism.
- **Security & Compliance**: Ensure PII filtering in retrieved content, secure API endpoints, and adhere to relevant data governance policies.

## Notes

- HTML generation remains disabled.