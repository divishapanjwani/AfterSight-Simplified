# References and how each was verified

Numbering follows first citation in the manuscript, as `build_paper.py` renumbers. Every DOI was resolved
through the Crossref API (`api.crossref.org/works/<doi>`) on 2026-10-25. Authors, title, venue, volume, issue,
pages and year were compared with that record. Entries without a DOI were checked on the publisher or
proceedings page.

| # | Reference | Verified via | Claim it supports in the paper |
|---|---|---|---|
| 1 | R. Gupta, B. Goodman, N. Patel, R. Hosfelt, S. Sajeev, E. Heim, J. Doshi, K. Lucas, H. Choset, M. Gaston, "Creating xBD: A dataset for assessing building damage from satellite imagery," Proc. IEEE/CVF CVPR Workshops, 2019, pp. 10–17. | CVF open-access listing (via search) and mlanthology BibTeX (author order, pages 10–17) | xBD dataset, four damage levels |
| 2 | R. Gupta *et al.*, "xBD: A dataset for assessing building damage from satellite imagery," arXiv:1911.09296, 2019. | arXiv / ar5iv (earlier session): "targeted … below a 0.8 meter ground sample distance" | the < 0.8 m GSD statement |
| 3 | R. Gupta (Rohit) and M. Shah, "RescueNet …," ICPR 2020 (pub. 2021), pp. 4405–4411, doi:10.1109/ICPR48806.2021.9412295 | Crossref | deep joint segmentation and damage models on xBD |
| 4 | Y. Shen, S. Zhu, T. Yang, C. Chen, D. Pan, J. Chen, L. Xiao, Q. Du, "BDANet …," IEEE TGRS, vol. 60, pp. 1–14, 2022, doi:10.1109/TGRS.2021.3080580 | Crossref (full 8-author list) | deep damage models on xBD |
| 5 | T. Cover, P. Hart, "Nearest neighbor pattern classification," IEEE Trans. Inf. Theory, 13(1):21–27, 1967, doi:10.1109/TIT.1967.1053964 | Crossref | KNN |
| 6 | L. Breiman, J. H. Friedman, R. A. Olshen, C. J. Stone, *Classification and Regression Trees*, Wadsworth, 1984; CRC reprint doi:10.1201/9781315139470 | Crossref (CRC reprint record) | CART decision trees, Gini |
| 7 | L. Dong, J. Shan, "A comprehensive review of earthquake-induced building damage detection with remote sensing techniques," ISPRS J. P&RS, 84:85–99, 2013, doi:10.1016/j.isprsjprs.2013.06.011 | Crossref | remote-sensing damage detection review |
| 8 | A. Singh, "Digital change detection techniques using remotely-sensed data," IJRS, 10(6):989–1003, 1989, doi:10.1080/01431168908903939 | Crossref | classical change detection |
| 9 | A. Cooner, Y. Shao, J. Campbell, "Detection of urban damage …: Revisiting the 2010 Haiti earthquake," Remote Sens., 8(10):868, 2016, doi:10.3390/rs8100868 | Crossref; abstract via search: texture and structure more important than spectral | classical ML and texture for damage |
| 10 | R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural features for image classification," IEEE TSMC, SMC-3(6):610–621, 1973, doi:10.1109/TSMC.1973.4309314 | Crossref | GLCM |
| 11 | J. Canny, "A computational approach to edge detection," IEEE TPAMI, PAMI-8(6):679–698, 1986, doi:10.1109/TPAMI.1986.4767851 | Crossref | Canny edges |
| 12 | T. Valentijn, J. Margutti, M. van den Homberg, J. Laaksonen, "Multi-hazard and spatial transferability of a CNN for automated building damage assessment," Remote Sens., 12(17):2839, 2020, doi:10.3390/rs12172839 | Crossref; **full text read** (Aalto repository PDF): 13 xBD disasters, AUC; "performance differed highly between occurrences of disasters" yet stayed high without test-disaster labels | cross-event generalisation (cited as a *contrast*, accurately) |
| 13 | F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," JMLR, 12:2825–2830, 2011 | JMLR paper page | implementation library |
| 14 | H. He, E. A. Garcia, "Learning from imbalanced data," IEEE TKDE, 21(9):1263–1284, 2009, doi:10.1109/TKDE.2008.239 | Crossref | accuracy dominated by the majority class |
| 15 | M. Sokolova, G. Lapalme, "A systematic analysis of performance measures for classification tasks," Inf. Process. Manage., 45(4):427–437, 2009, doi:10.1016/j.ipm.2009.03.002 | Crossref | macro-averaged measures |
| 16 | B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall/CRC, 1994 (DOI record year), doi:10.1201/9780429246593 | Crossref | bootstrap confidence intervals |
| 17 | C. A. Field, A. H. Welsh, "Bootstrapping clustered data," JRSS B, 69(3):369–390, 2007, doi:10.1111/j.1467-9868.2007.00593.x | Crossref | resampling whole clusters (scenes) |

## Notes

- **Efron & Tibshirani** is often cited as 1993 (first printing); the DOI record lists 1994. The paper uses
  1994 to match the DOI.
- **Breiman *et al.***: the original 1984 Wadsworth edition is cited, with the CRC reprint DOI for access.
- **Ref. 1 and ref. 3** have different first authors with the same initial: Ritwik Gupta (xBD) and Rohit
  Gupta (RescueNet).
- **Corrections:** BDANet's author list, Cooner's initials and the description of Valentijn *et al.* were
  corrected during verification.
