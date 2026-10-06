# Summary of camera-ready changes — D2AI 2026, DM2158

**Mining Attention Dynamics: Auditing KV-Saliency Prediction in LLMs**

We thank the reviewer for the four concerns below. The revised manuscript treats
KVSalienceBench as a measurement and benchmark contribution. This summary describes
the revision; it does not claim new GPU experiments during the final editorial pass.

**1. Attention masking versus physical eviction.** The abstract, §VII, §IX and
conclusion scope answer-quality findings to the evaluated masked loop. Logical
retention is distinguished from physical storage and serving efficiency. The
separate physical-KV reference reports storage and parity checks; extended
mask/physical tests exceed the preset numerical tolerance. No simulator or
allocator measurement supports a production per-token-latency or throughput claim.

**2. Offline predictor versus runtime scorer.** §IV-B and Table I specify the
different feature realizations, and Exp#8 explicitly identifies its policy as the
runtime reconstruction rather than the offline predictor of Table II. The bridge
compares feature realizations on matched rows and separately along an on-policy
trajectory. Changing trajectory and candidate set together does not identify the
cause of the historical proxy gap. The draw.io overview shows the distinct
offline, masked-loop and physical-reference paths.

**3. Scope of query–key redundancy.** §IX discloses the legacy query/key coordinate
mismatch and withdraws the “faithful Quest” interpretation. Version-2 phase-aligned
token/head dot-max controls are retained and have small, opposite-signed changes
in the two tested model configurations. They are neither Quest's page-bound
algorithm nor evidence of intrinsic query redundancy. The artifact keeps the synthetic needle experiment as a negative control
(`docs/HISTORICAL_DIAGNOSTICS.md`).

**4. External validity.** §IX separates the main two-model evidence from historical
workload, 14B and 16K extensions. Findings do not generalize to MoE, base models,
contexts of at least 32K, or temperature sampling. Exp#8 reports mean equivalence
within a specified tolerance on seven QA datasets, with task heterogeneity visible
in the forest plot; both compressed policies lose quality relative to full cache.

The introduction and conclusion identify versioned collection, provenance and an
explicit benchmark protocol as contributions to data and database systems for AI.
The README and English presentation lead with the same v2 evidence as the manuscript.
Historical results remain labeled and unchanged.
