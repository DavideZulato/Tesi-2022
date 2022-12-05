# Tesi-2022: Discrimination in HR Analytics. A fair Workflow
Codici e materiale per la tesi "Discrimination in HR Analytics. A fair Workflow" A.A. 2021/2022 riguardante l'analisi del BIAS nei predictive algorithms in recruitment e HRM utilizzando un test econometrico/statistico per verificare se un algoritmo discrimina un gruppo specifico (S, $sensitive feature$) e agire con tecniche di pre-processing (e.g reweighting) o durante il training dei modelli per ridurre l' $unfairness$ nei confronti della classe protetta. L'attenzione è rivolta alla comprensione e all'attenuazione della discriminazione basata su caratteristiche sensibili.

# Workflow

Generazione dei dati (BIAS nei dati) $\rightarrow$ Fit modelli $\rightarrow$ XAI $\rightarrow$ BIAS mitigation  
Classificatore $\hat{Y}=f(X,S)$ dove $S$ è la feature protetta e $Y \in [0,1]$
# Files

- Simulating data.R contiene la simulazione del dataset utilizzato in contesto HRM costruito per discriminare la classe $S_d \in S$ (classe svantaggiata della feature sensibile $S$ )

- LR_RF_SVM_XGBoost.R contiene training dei modelli, performances in training e test set, Model explanation locale e globale, fairness check per i modelli: logistic regression, Random forest, SVM e XGBoost.

- Synthetic_data 28_11_22.html contiene un workflow dalla generazione dei dati ai counterfactuals con modelli Random Forest, SVM, logistic with splines e gbm con pesi custom per agire su alcune misure di unfairness.

- Matching.R per identificare (con pacchetto matchit) un impatto causale della feature sensibile $S$ sulla probabilità di ottenere un outcome positivo creando un counterfactual group con observational data. 

- Reweighting the data.R per costruire i pesi custom sulla base del codice python in Responsible AI (https://link.springer.com/book/10.1007/978-3-030-76860-7)

- Fairness_sim_data GBM.R contiene un'analisi del dataset simulato con modello gbm e BIAS mitigation. si utilizza il pacchetto fairmodels e gradient boosting (https://cran.r-project.org/web/packages/gbm/gbm.pdf) con i seguenti steps: 1)fit the model 2)create global and local explanations 3) Bias mitigation. L'analisi è basata sul lavoro di Jakub Wiśniewski: fairmodels advanced Tutorial (https://modeloriented.github.io/fairmodels/articles/Advanced_tutorial.html)
