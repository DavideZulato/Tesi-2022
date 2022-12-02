# Tesi-2022: A fair workflow in HR analyics
Code for the work "A fair workflow in HR analyics"

# Files
- LR_RF_SVM_XGBoost.R contiene training dei modelli, performances in training e test set, Model explanation locale e globale, fairness check per i modelli: logistic regression, Random forest, SVM e XGBoost.

- Synthetic_data 28_11_22.html contiene un workflow dalla generazione dei dati ai counterfactuals con modelli Random Forest, SVM, logistic with splines e gbm con pesi custom per agire su alcune misure di unfairness.

- Matching.R per identificare (con pacchetto matchit) un impatto causale della feature sensibile $S$ sulla probabilità di ottenere un outcome positivo creando un counterfactual group con observational data. 

- Reweighting the data.R per costruire i pesi custom sulla base del codice python in Responsible AI (https://link.springer.com/book/10.1007/978-3-030-76860-7)
