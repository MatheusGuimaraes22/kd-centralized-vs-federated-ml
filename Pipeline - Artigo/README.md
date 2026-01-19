# Pipeline - Artigo

Esta pasta contem os scripts usados no pipeline do artigo (sem scripts de plot/tabelas/figuras).

## Conteudo
- ckd_make_dataset.py: preprocessamento do dataset (limpeza, imputacao, one-hot, normalizacao, holdout)
- feature_selection.py: definicao de cenarios e carga de colunas de drop
- chi2_select_drop.py: selecao de atributos via chi-square (top-k)
- ckd_paper_kfold.py: loader e preprocessor base
- central_holdout_eval.py: avaliacao centralizada (holdout)
- hparam_search.py: otimizacao de hiperparametros (CV)
- fl_simulate_scenarios_kfold.py: CV para modelos FL por cenarios
- fl_perturbation_iid_vs_noniid_cv.py: CV FL IID vs non-IID (Dirichlet)
- fl_oof_predictions_cv.py: predicoes OOF (CV) para central e FL
- fl_holdout_predictions_and_calibration.py: predicoes holdout para FL (sem plots)
- fl_round_metrics.py: metricas por rodada (convergencia)
- bootstrap_oof_metric_comparison.py: bootstrap + DeLong com OOF
- bootstrap_holdout_metric_comparison.py: bootstrap + DeLong com holdout
- bootstrap_iid_vs_noniid_cv.py: bootstrap de metricas por fold (IID vs non-IID)
