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

## Addendum, 2026-10-05

The table above is the 24-entry verification of the original submission and is kept as it
was. The camera-ready bibliography compiles 27 entries. Two records were re-checked on
2026-10-05:

| key | record (verified) | evidence |
|---|---|---|
| zhou2018din (new, [14]) | G. Zhou, X. Zhu, C. Song, Y. Fan, H. Zhu, X. Ma, Y. Yan, J. Jin, H. Li, K. Gai, "Deep interest network for click-through rate prediction," Proc. 24th ACM SIGKDD, 2018, pp. 1059--1068 | Crossref, DOI 10.1145/3219819.3219823; the user-weighted AUC it is cited for is defined in its Section 6.3 |
| peng2005mrmr ([13]) | title printed with a colon: "...mutual information: Criteria of max-dependency, max-relevance, and min-redundancy" | PubMed PMID 16119262 prints the colon; Crossref's deposited metadata (DOI 10.1109/TPAMI.2005.159) does not. Volume 27, number 8, pages 1226--1238, 2005 agree in both |

## Addendum, 2026-10-06 (second reference audit)

An external field-by-field audit of the 27 printed references (2026-10-05) found no wrong
paper, no missing author and no wrong venue, and asked for three field changes, all
applied. The camera-ready bibliography now compiles **28 entries**: LightGBM is cited at
its first mention as [8], so entries [8]-[27] of the audited PDF are [9]-[28].

| key | change | evidence |
|---|---|---|
| zhou2018din ([15]) | `Song, Chenru` -> `Song, Chengru`; author order unchanged; the PDF prints "C. Song" either way | The KDD 2018 accepted-paper page and arXiv 1706.06978 print "Chengru Song"; Crossref's deposited record for DOI 10.1145/3219819.3219823 prints "Chenru Song". The sources also differ in author order; the entry keeps the order of the proceedings record |
| agrawal2026benchmarking ([23]) | `doi` field holds the identifier without TeX escapes (`10.15495/EPub_UBT_00009365`) | `IEEEtran.bst` prints the URL, not the DOI field, so nothing visible changes |
| kvpress ([25]) | explanatory `note` removed; the entry ends "arXiv preprint arXiv:2510.00636, 2025." | The KVPress relation is now stated in the related-work text. arXiv abstract checked 2026-10-06: Expected Attention "estimates KV pairs importance by predicting how future queries will attend to them" and releases KVPress "already including more than 20 techniques" |
| ke2017lightgbm (new, [8]) | G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," Proc. Advances in Neural Information Processing Systems, vol. 30, 2017 | proceedings.neurips.cc record and its BibTeX (checked 2026-10-06). The official BibTeX has an empty `pages` field and page ranges were not verified against a second source, so none is printed |

Still open from the audit: the page range of AttentionPredictor ([7], 153646-153678) is
kept as printed and has not been confirmed from an official proceedings export.
