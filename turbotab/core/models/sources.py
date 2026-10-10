"""The sources a model family's declarations may cite, by key (MODEL_FAMILY_CONTRACT C5, §7).

A :class:`~turbotab.core.models.base.Source` names one of these keys, and ``register_family``
refuses a key that is not here. Each entry is a verified reference listed in MODEL_FAMILY_CONTRACT §7
or in RECIPES_AND_TUNING's Sources; a source joins only once one of them lists it, and only when a
declaration cites it. A long author list is written "First, A., Second, B., et al."
Each key is also a record of SIZING X4's citation registry (:mod:`turbotab.core.export.citations`,
every DOI checked against Crossref), under the same key; ``test_citations`` holds them together.
"""
from __future__ import annotations

SOURCES: dict[str, str] = {
    "friedman2001": "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting "
                    "machine. Annals of Statistics 29(5). https://doi.org/10.1214/aos/1013203451",
    "hastie2009": "Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical "
                  "Learning, 2nd ed. Springer. §3.4.1, eqs. 3.47 and 3.50. "
                  "https://hastie.su.domains/ElemStatLearn/",
    "huber1964": "Huber, P. J. (1964). Robust estimation of a location parameter. Annals of "
                 "Mathematical Statistics 35(1):73–101. https://doi.org/10.1214/aoms/1177703732",
    "holland1977irls": "Holland, P. W., Welsch, R. E. (1977). Robust regression using iteratively "
                       "reweighted least-squares. Communications in Statistics - Theory and Methods "
                       "6(9):813–827. https://doi.org/10.1080/03610927708827533",
    "breiman2001forests": "Breiman, L. (2001). Random Forests. Machine Learning 45(1):5–32. "
                          "https://doi.org/10.1023/A:1010933404324",
    "liaw2002randomforest": "Liaw, A., Wiener, M. (2002). Classification and Regression by randomForest. R "
                            "News 2(3):18–22. https://CRAN.R-project.org/doc/Rnews/Rnews_2002-3.pdf",
    "wright2017ranger": "Wright, M. N., Ziegler, A. (2017). ranger: A Fast Implementation of Random "
                        "Forests for High Dimensional Data in C++ and R. Journal of Statistical Software "
                        "77(1). https://doi.org/10.18637/jss.v077.i01",
    "probst2019rf": "Probst, P., Wright, M. N., Boulesteix, A.-L. (2019). Hyperparameters and tuning "
                    "strategies for random forest. WIREs Data Mining and Knowledge Discovery "
                    "9(3):e1301. https://doi.org/10.1002/widm.1301",
    "probst2019tunability": "Probst, P., Boulesteix, A.-L., Bischl, B. (2019). Tunability: Importance of "
                            "Hyperparameters of Machine Learning Algorithms. Journal of Machine Learning "
                            "Research 20(53):1–32. https://jmlr.org/papers/v20/18-444.html",
    "chen2016xgboost": "Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
                       "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
                       "Discovery and Data Mining, pp. 785–794. https://doi.org/10.1145/2939672.2939785",
    "bergstra2012random": "Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. "
                          "Journal of Machine Learning Research 13:281–305. "
                          "https://jmlr.org/papers/v13/bergstra12a.html",
    "bischl2023hpo": "Bischl, B., Binder, M., Lang, M., et al. (2023). Hyperparameter optimization: "
                     "Foundations, algorithms, best practices, and open challenges. WIREs Data Mining "
                     "and Knowledge Discovery 13(2):e1484. https://doi.org/10.1002/widm.1484",
    "cawley2010overfitting": "Cawley, G. C., Talbot, N. L. C. (2010). On Over-fitting in Model Selection and "
                             "Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning "
                             "Research 11:2079–2107. https://jmlr.org/papers/v11/cawley10a.html",
    "riley2021penalization": "Riley, R. D., Snell, K. I. E., Martin, G. P., et al. (2021). Penalization and "
                             "shrinkage methods produced unreliable clinical prediction models especially when "
                             "sample size was small. Journal of Clinical Epidemiology 132:88–96. "
                             "https://doi.org/10.1016/j.jclinepi.2020.12.005",
    "vancalster2020shrinkage": "Van Calster, B., van Smeden, M., De Cock, B., Steyerberg, E. W. (2020). "
                               "Regression shrinkage methods for clinical prediction models do not guarantee "
                               "improved performance: Simulation study. Statistical Methods in Medical Research "
                               "29(11):3166–3178. https://doi.org/10.1177/0962280220921415",
    "martin2021tuning": "Martin, G. P., Riley, R. D., Collins, G. S., Sperrin, M. (2021). Developing "
                        "clinical prediction models when adhering to minimum sample size recommendations: "
                        "The importance of quantifying bootstrap variability in tuning parameters and "
                        "predictive performance. Statistical Methods in Medical Research "
                        "30(12):2545–2561. https://doi.org/10.1177/09622802211046388",
    "kruppa2014theory": "Kruppa, J., Liu, Y., Biau, G., et al. (2014). Probability estimation with "
                        "machine learning methods for dichotomous and multicategory outcome: Theory. "
                        "Biometrical Journal 56(4):534–563. https://doi.org/10.1002/bimj.201300068",
    "josse2024missing": "Josse, J., Chen, J. M., Prost, N., Scornet, E., Varoquaux, G. (2024). On the "
                        "consistency of supervised learning with missing values. Statistical Papers "
                        "65(9):5447–5479. https://doi.org/10.1007/s00362-024-01550-4",
    "perezlebel2022missing": "Perez-Lebel, A., Varoquaux, G., Le Morvan, M., Josse, J., Poline, J.-B. (2022). "
                             "Benchmarking missing-values approaches for predictive models on health "
                             "databases. GigaScience 11:giac013. https://doi.org/10.1093/gigascience/giac013",
    "vanness2023indicator": "Van Ness, M., Bosschieter, T. M., Halpin-Gregorio, R., Udell, M. (2023). The "
                            "Missing Indicator Method: From Low to High Dimensions. Proceedings of the 29th "
                            "ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 5004–5015. "
                            "https://doi.org/10.1145/3580305.3599911",
    "gelman2008twosd": "Gelman, A. (2008). Scaling regression inputs by dividing by two standard "
                       "deviations. Statistics in Medicine 27(15):2865–2873. "
                       "https://doi.org/10.1002/sim.3107",
    "ng2004l1l2": "Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational "
                  "invariance. Proceedings of the 21st International Conference on Machine Learning "
                  "(ICML 2004), p. 78. https://doi.org/10.1145/1015330.1015435",
    "mentch2020randomization": "Mentch, L., Zhou, S. (2020). Randomization as Regularization: A Degrees of "
                               "Freedom Explanation for Random Forest Success. Journal of Machine Learning "
                               "Research 21(171):1–36. https://jmlr.org/papers/v21/19-905.html",
    "mcelfresh2023tabular": "McElfresh, D., Khandagale, S., Valverde, J., et al. (2023). When Do Neural Nets "
                            "Outperform Boosted Trees on Tabular Data? Advances in Neural Information "
                            "Processing Systems 36 (Datasets and Benchmarks), pp. 76336–76369. "
                            "https://doi.org/10.52202/075280-3337",
    "kobak2020ridge": "Kobak, D., Lomond, J., Sanchez, B. (2020). The Optimal Ridge Penalty for "
                      "Real-world High-dimensional Data Can Be Zero or Negative due to the Implicit "
                      "Ridge Regularization. Journal of Machine Learning Research 21(169):1–16. "
                      "https://jmlr.org/papers/v21/19-844.html",
    "curth2024forests": "Curth, A., et al. (2024). Why do Random Forests Work? "
                        "Understanding Tree Ensembles as Self-Regularizing Adaptive Smoothers. arXiv "
                        "preprint. https://doi.org/10.48550/arXiv.2402.01502",
}

__all__ = ["SOURCES"]
