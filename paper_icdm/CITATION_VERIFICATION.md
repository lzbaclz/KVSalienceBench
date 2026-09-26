# Phase F — Citation web-verification report (HARD GATE) — PASS

24 cited bib entries, each independently web-verified (title + first author + venue + year,
and arXiv-id->paper match) against official proceedings / DBLP / ACL Anthology / arXiv.

## Verdict: ALL 24 = OK. No fabrication, no arXiv mismatch, no venue/year error.

| key | venue/year (verified) | evidence |
|---|---|---|
| h2o | NeurIPS 2023 | NeurIPS proc + arXiv 2306.14048 |
| streamingllm | ICLR 2024 | ICLR proc/OpenReview (arXiv 2309.17453) |
| snapkv | NeurIPS 2024 | NeurIPS proc + arXiv 2404.14469 |
| pyramidkv | arXiv 2406.02069 (2024) | arXiv + MSR pub page |
| adakv | NeurIPS 2025 | NeurIPS 2025 poster + arXiv 2407.11550 |
| nacl | ACL 2024 (long) | ACL Anthology 2024.acl-long.428 + arXiv 2408.03675 |
| locret | TMLR 2025 | OpenReview "Accepted by TMLR" + arXiv 2410.01805 |
| quest | ICML 2024 | PMLR v235 / DBLP + arXiv 2406.10774 |
| infinigen | OSDI 2024 | USENIX OSDI'24 + arXiv 2406.19707 |
| duoattention | ICLR 2025 | ICLR proc/OpenReview + arXiv 2410.10819 |
| tactic | arXiv 2502.12216 (2025) | arXiv/DBLP (no peer venue) |
| pagedattention | SOSP 2023 | ACM DOI 10.1145/3600006.3613165 + arXiv 2309.06180 |
| longbench | ACL 2024 (long) | ACL Anthology 2024.acl-long.172 + arXiv 2308.14508 |
| mooncake | USENIX FAST 2025 (Best Paper) | USENIX FAST'25 + arXiv 2407.00079 |
| blummitchell | COLT 1998 | DBLP + ACM DL |
| gama2014drift | ACM Comput. Surv. 46(4) 2014 | ACM DOI 10.1145/2523813 |
| vfdt | KDD 2000 | ACM DOI 10.1145/347090.347107 |
| gibbs2021aci | NeurIPS 2021 | NeurIPS proc + arXiv 2106.00170 |
| vovk2005algorithmic | Springer 2005 (1st ed.) | Springer/ISBN 978-0387001524 |
| guo2017calibration | ICML 2017 | PMLR v70 pp.1321-1330 |
| mcgill1954multivariate | Psychometrika 19(2) 1954 | RePEc/Springer |
| williams2010nonnegative | arXiv 1004.2515 (2010) | arXiv |
| peng2005mrmr | IEEE TPAMI 27(8) 2005 | IEEE Xplore |
| angelopoulos2023gentle | Found. Trends ML 2023 | arXiv 2107.07511 / FnT |

## Orphan bib entries (defined, NOT cited — harmless, excluded from compiled bib)
bolukbasi2017adaptive, elkan2001foundations, ftrl, halo, orchkv, ruler, seer.
- `ruler` (RULER benchmark) is a candidate to CITE in the multi-key negative-control paragraph.
- `seer`/`halo`/`orchkv` look like own-work keys: keep out of the cited set under triple-blind;
  retain in bib only for de-anonymized camera-ready self-citation if desired.

Method: 24 parallel web-verifier agents (workflow citation-verify), 108 web tool-uses.
