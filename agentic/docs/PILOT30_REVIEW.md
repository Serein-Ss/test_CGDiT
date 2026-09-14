# 30 条试点核查结果（第二阶段）

本表是助手来源审查，不是独立金标准。全部记录暂不可训练；“来源可用”仅表示取得片段。原始数值保持不变。

| 记录 | 分组 | 原始 K | 事件审查 | 来源 | 主要问题 |
|---|---|---:|---|---|---|
| 0.782_NdScO3 | test | 953.0 | unknown | [来源](https://www.sciencedirect.com/science/article/pii/S0921452696010678) | 0.953 K conflicts with 953 K. Symbol Tc does not establish a Curie event. |
| 0.639_Mn2Au | train | 1000.0 | neel | [来源](https://www.nature.com/articles/s41467-023-41569-z) | 1000 is a lower bound in this background statement, not an exact Curie value. Original measurement/extrapolation must be traced. |
| 0.108_Mn3Ir | train | 960.0 | neel | [来源](https://www.nature.com/articles/ncomms3892) | Cited comparison compound, not the material measured in this paper; trace reference 14. |
| 0.186_CeMnAsO | train | 347.0 | neel | [来源](https://impact.ornl.gov/en/publications/spin-reorientation-and-ce-mn-coupling-in-antiferromagnetic-oxypni/) | Previously sourced diagnostic; requires independent review and structure matching. |
| 0.187_CeMnAsO | train | 35.0 | spin_reorientation_and_neel | [来源](https://impact.ornl.gov/en/publications/spin-reorientation-and-ce-mn-coupling-in-antiferromagnetic-oxypni/) | Previously sourced diagnostic; requires independent review and structure matching. |
| 0.188_CeMnAsO | train | 7.0 | other | [来源](https://impact.ornl.gov/en/publications/spin-reorientation-and-ce-mn-coupling-in-antiferromagnetic-oxypni/) | Previously sourced diagnostic; requires independent review and structure matching. |
| 0.426_EuMnBi2 | train | 315.0 | neel | [来源](https://link.aps.org/accepted/10.1103/PhysRevLett.122.127207) | Previously sourced diagnostic; requires independent review and structure matching. |
| 2.51_EuMnBi2 | train | 19.0 | neel | [来源](https://arxiv.org/abs/1407.6203) | Source reports near 22 K, not local 19 K. Cross-sample discrepancy, not an automatic correction. |
| 0.65_Fe2O3-alpha | test | 955.0 | neel | [来源](https://www.sciencedirect.com/science/article/abs/pii/S0304885321007666) | Hematite has canted AFM order and a weak net moment; do not equate weak ferromagnetism with the target FM/FiM class. |
| 0.274_Mn4N | train | 745.0 | curie_candidate | [来源](https://tsukuba.repo.nii.ac.jp/record/47579/file_preview/JCG_489.pdf) | Previously sourced diagnostic; requires independent review and structure matching. |
| 1.495_YMn2Si2 | train | 460.0 | neel | [来源](https://www.sciencedirect.com/science/article/pii/0038109881905329) | Temperature must be linked by respective order; 460 belongs to YMn2Si2. |
| 1.689_LuMn2Ge2 | train | 508.0 | unknown | 待补原始证据 | Bounded searches performed; only indirect/related-compound leads found. Original MAGNDATA page unavailable. No label inferred. |
| 1.253_CeCo2P2 | train | 440.0 | neel | [来源](https://www.sciencedirect.com/science/article/pii/S0925838897004866) | Preserve 440(5) K, assign Co sublattice only where source context supports it. |
| 0.1024_BaFe2Se4 | test | 310.0 | neel | [来源](https://www.osti.gov/servlets/purl/1772614) | Canted AFM, approximate 310 K; spontaneous moment does not establish a conventional ferromagnet. |
| 1.281_YBaCuFeO5 | train | 446.0 | neel | [来源](https://www.sciencedirect.com/science/article/abs/pii/003810989090535J) | Paper concerns oxygen-nonstoichiometric phase; delta and Fe/Cu occupancy must match before using local structure. |
| 0.836_DyFeO3 | train | 650.0 | neel | [来源](https://advanced.onlinelibrary.wiley.com/doi/10.1002/admi.202400938) | Bulk background 645 K differs from local 650 K; article also fixes 650 K as a fit parameter. Do not treat a fixed parameter as a fresh measurement. |
| 0.619_LaMnAsO | test | 317.0 | unknown | [来源](https://link.springer.com/article/10.1186/s40679-016-0019-9) | 317 K is explicitly linked to MnAs impurity, NOT LaMnAsO. Quarantine target value, do not produce a negative Tc label. |
| 0.802_CuFeS2 | train | 823.0 | neel | [来源](https://www.researchgate.net/publication/345127292_Low-temperature_synthesis_of_micro-_and_nano-crystalline_CuFeS2_polymorphs) | Original paper background on chalcopyrite phase, surfaced through indexed paper text; bulk/phase correspondence not verified. |
| 0.279_Mn3As | train | 420.0 | unknown | 待补原始证据 | Bounded searches performed; only indirect/related-compound leads found. Original MAGNDATA page unavailable. No label inferred. |
| 1.420_YBa2Cu3O6 | train | 420.0 | neel | [来源](https://www.ncnr.nist.gov/staff/jeff/NCCOprb68%20144503%202003.pdf) | Short background excerpts lack full value-order context. Use as retrieval lead; original YBa2Cu3O6 measurement needed. |
| 0.617_KMnSb | test | 295.0 | unknown | [来源](https://www.cryst.ehu.es/magndata/index.php?this_label=0.617) | Exact MAGNDATA record ID matched. Transition upper bound lost in CSV. Experiment temperature is NOT an event. Magnetic type not explicit in excerpt. |
| 1.292_HoNi2B2C | train | 2.0 | unknown | [来源](https://www.sciencedirect.com/science/article/abs/pii/S0304885300009756) | Local 2 K not established as a transition; possible measurement/context mismatch. Do not replace it by 5 or 6 K without original record. |
| 0.909_Er2PtGe6 | val | 9.0 | neel | [来源](https://ruj.uj.edu.pl/server/api/core/bitstreams/dda3ca05-1a5e-4955-a8dd-f3859cee9c2b/content) | Original paper reports 4.9 K for Er2PtGe6; local 9 K is a conflict requiring original record correspondence. |
| 0.36_NiF2 | val | 73.0 | neel | [来源](https://www.sciencedirect.com/science/article/pii/S0304885306018324) | Canted rutile antiferromagnet; target event is not Curie. |
| 0.327_CsMnF4 | val | 9.0 | unknown | [来源](https://www.sciencedirect.com/science/article/abs/pii/0022459680905599) | Ferromagnetic structure supported; this excerpt does not establish 9 K or the transition itself. Curie candidate pending temperature evidence. |
| 2.11_TbMg | train | 81.0 | unknown | 待补原始证据 | Bounded searches performed; only indirect/related-compound leads found. Original MAGNDATA page unavailable. No label inferred. |
| 1.314_NaFeSi2O6 | train | 8.0 | other | [来源](https://kups.ub.uni-koeln.de/5258/) | Primary experimental thesis abstract: 8 K spin-density wave onset, 6 K helix; natural versus synthetic samples differ. |
| 1.0.59_Na2MnTeO6 | val | 5.0 | neel | [来源](https://oiks.pnpi.spb.ru/articles/953) | P-31c (163) phase in source consistent with local SG, but coordinates not matched. 3R R-3 polytype is distinct; do not transfer its absence of order. |
| 1.775_CaCu3Ti4O12 | train | 27.0 | neel | [来源](https://www.sciencedirect.com/science/article/abs/pii/S0038109802000613) | Single crystal source gives 24.1(1) K, while older powder work may give 27 K. Preserve sample-dependent discrepancy. |
| 1.68_NaNdFeWO6 | train | 25.0 | neel | [来源](https://www.ornl.gov/publication/polar-and-magnetic-layered-site-and-rock-salt-b-site-ordered-nalnfewo6-ln-la-nd) | 25 K belongs to La compound in this paper, ~21 K to Nd. Potential cross-compound attribution error; retain local value pending record match. |
