# Pipeline de rehaussement et de segmentation vasculaire

Ce dépôt contient le pipeline expérimental utilisé pour étudier l'influence de différents opérateurs de dérivation dans le calcul du Hessien, puis leur impact sur le rehaussement et la segmentation de structures vasculaires 3D.

Le dépôt regroupe quatre niveaux d'expérimentation :

1. **prétraitement des données** ;
2. **benchmark principal du pipeline Hessien → vesselness → segmentation** ;
3. **analyses post-benchmark** (composantes connexes, faux positifs, bifurcations, statistiques, etc.) ;
4. **études analytiques sur fantôme synthétique** permettant d'évaluer séparément l'approximation du Hessien, les valeurs propres, la vesselness et la segmentation.

> **Important — lecture du dépôt**
>
> Le dossier `core/` contient le pipeline principal actuellement utilisé pour les benchmarks. Les dossiers `core/Hessian_evaluation/`, `Manipulation_des_donnees/`, `docs/` et `visualisation_résultats/` regroupent principalement les études auxiliaires, l'exploration/préparation des données et les outils de visualisation. Ils ne sont pas tous nécessaires pour exécuter le benchmark principal.
 ATTENTION: Pour lancer une dataset modifiez le chemain, l'échelle appropriée.. 
---

## 1. Vue d'ensemble

Le chemin principal d'une expérience est :

Données prétraitées
        │
        ▼
PatientLoader / Loader
        │
        ▼
Processor
        │
        ├── Derivator
        │      └── calcul du Hessien 3D
        │
        ├── Enhancer
        │      ├── Frangi
        │      ├── Jerman
        │      └── MFAT
        │
        └── Segmenter
               └── seuillage / optimisation du seuil
        │
        ▼
Métriques
        │
        ├── Dice
        ├── MCC
        ├── ROC AUC
        ├── PR AUC
        ├── clDice
        ├── composantes connexes
        ├── connectivité du squelette
        └── bifurcations
        │
        ▼
BenchmarkHessian / AnalyticsRunner
        │
        ├── tableaux
        ├── figures
        ├── JSON
        └── résultats par patient


---

# 2. Où trouver rapidement quelque chose ?

Cette section est destinée à permettre à quelqu'un qui reprend le projet de retrouver immédiatement le code responsable d'un résultat.

| Ce que l'on cherche | Fichier principal | Fonction / classe |
|---|---|---|
| Point d'entrée du projet | `main.py` | `main()` |
| Arguments de la ligne de commande | `configs/args.py` | `get_parser()` |
| Chargement des patients | `core/utils/patient_loader.py` | `PatientLoader` |
| Chargement générique | `core/io/loader.py` | `Loader` |
| Orchestration du pipeline | `core/processing/processor.py` | `Processor` |
| Calcul du Hessien | `core/processing/derivator.py` | `Derivator` |
| Frangi / Jerman / MFAT | `core/processing/enhancer.py` | `Enhancer` |
| Seuillage / segmentation | `core/processing/segmenter.py` | `Segmenter` |
| Métriques | `core/experiments/metrics.py` | fonctions de métriques |
| Benchmark principal | `core/experiments/benchmarks/hessian.py` | `BenchmarkHessian` |
| Exécution du benchmark | `core/experiments/benchmarks/runner.py` | `BenchmarkRunner` |
| Tableau moyen ± écart-type | `core/experiments/analytics/runner.py` | `_generate_mean_std_table()` |
| Tableau patient par patient | `core/experiments/analytics/runner.py` | `_generate_patient_table()` |
| Figures principales du benchmark | `core/experiments/analytics/runner.py` | `_create_overview_metrics()`, `_create_overview_patients()` |
| Statistiques vesselness | `core/experiments/analytics/runner.py` | `get_enhanced_stats_table()` |
| Matrices de confusion | `core/experiments/benchmarks/hessian.py` | `_log_and_store_confusion()`, `_create_confusion_summary_figure()` |
| `confusion_matrices.json` | `core/experiments/benchmarks/hessian.py` | données enregistrées pendant le benchmark |
| Statistiques FP | `core/experiments/benchmarks/hessian.py` | `_fp_statistics()`, `_save_fp_stats_incremental()` |
| Analyse des bifurcations | `core/experiments/benchmarks/hessian.py` | `analyze_bifurcations()` |
| Analyse des prédictions | `core/experiments/benchmarks/hessian.py` | `analyze_prediction_data()` |
| Sauvegarde des résultats | `core/io/saver.py` | `Saver` |
| Étude Hessienne analytique | `core/Hessian_evaluation/Comparaison_hessian.py` | `run_study()` / `run_benchmark()` |
| Étude des valeurs propres | `core/Hessian_evaluation/Comparaison_valeurs_propres.py` | `run_study()` / `run_benchmark()` |
| Étude vesselness | `core/Hessian_evaluation/Comparaison_vesselness.py` | `run_study()` / `run_benchmark()` |
| Étude segmentation synthétique | `core/Hessian_evaluation/Comparaison_segmentation.py` | `run_study()` / `run_benchmark()` |
| Filtrage par composantes connexes | `core/experiments/post_benchmark/cc_filtering/` | `run_cc_filtering_study()` |
| Tests post-benchmark | `core/experiments/post_benchmark/tests/post_benchmark_tests.py` | `run_post_benchmark_tests()` |
| Prétraitement multi-dataset | `Manipulation_des_donnees/core_pre_traitement/multi_dataset_preprocessor.py` | `MultiDatasetPreprocessor` |
| Exploration IRCAD/Bullitt/VascuSynth | `docs/` | notebooks |
| Visualisation squelette | `visualisation_résultats/Visualize_skeleton_consensus.py` | fonctions de visualisation |
| Visualisation volumique 3D | `visualisation_résultats/visualiser_volumique3D.py` | fonctions de visualisation |
| Visualisation matrices de confusion | `visualisation_résultats/visualiser_cv.py` | fonctions de visualisation |

---

# 3. Arborescence du dépôt

pipeline-vessel-enhancement-master/
│
├── main.py
├── requirements.txt
├── gpu_env.yaml
├── .gitignore
│
├── configs/
│   ├── args.py
│   ├── benchmark/
│   ├── operator_study/
│   ├── post_benchmark_studies/
│   └── preprocessing/
│
├── core/
│   ├── config/
│   ├── processing/
│   ├── experiments/
│   ├── io/
│   ├── utils/
│   └── Hessian_evaluation/
│
├── Manipulation_des_donnees/
│   ├── core_exploration/
│   ├── core_pre_traitement/
│   └── rechargement_des_donnees.py
│
├── docs/
│   ├── README.md
│   ├── Exploration_données IRCAD/
│   ├── Exploration_données Bullitt/
│   └── Vascu-synth/
│
└── visualisation_résultats/

---

# 4. Point d'entrée : `main.py`

`main.py` est le point d'entrée principal de toutes les expériences exposées par la ligne de commande.

Il peut lancer :

- le prétraitement ;
- le benchmark principal :
- les études analytiques des opérateurs ;
- les analyses post-benchmar: tests statistiques, le filtrage par composantes connexe: (uniquement si on a les résultats du benchmark principal)

Les fonctions principales sont chargées dynamiquement selon les arguments fournis.

## Commandes principales

### Benchmark principal

```bash
python3 main.py --run_benchmark --benchmark_type hessian --dataset ircad --enhancer frangi
```

Pour MFAT :

```bash
python3 main.py --run_benchmark --benchmark_type hessian --dataset ircad --enhancer mfat
```

Pour Jerman :

```bash
python3 main.py --run_benchmark --benchmark_type hessian --dataset ircad --enhancer jerman
```

### Prétraitement

```bash
python3 main.py --run_preprocessing --preprocess_dataset ircad
```

Valeurs disponibles dans l'argument CLI : `ircad`, `bullitt`, `vascusynth`, `both`, `all`.

### Étude analytique du Hessien

```bash
python3 main.py --run_operator_study --study_type hessian
```

### Étude des valeurs propres

```bash
python3 main.py --run_operator_study --study_type eigenvalues
```

### Étude de la vesselness

```bash
python3 main.py --run_operator_study --study_type vesselness
```

### Étude de segmentation synthétique

```bash
python3 main.py --run_operator_study --study_type segmentation
```

### Analyse post-benchmark

```bash
python3 main.py --run_post_tests --dataset ircad
```

### Filtrage par composantes connexes

```bash
python3 main.py --run_cc_filtering --dataset ircad
```

---

# 5. Configuration du benchmark principal

Les fichiers importants sont dans :

```text
configs/benchmark/
```

## `configs/benchmark/runner.yaml`

Contrôle l'environnement général d'exécution :

- nom de l'expérience ;
- dossier d'entrée ;
- dossier de sortie ;
- logs ;
- affichage des figures ;
- sauvegarde ;
- dossiers `images`, `labels` et `masks` ;
- sélection éventuelle des patients.

Les champs principaux sont :

```yaml
setup:
  name: "hessian"
  input_dir: "..."
  output_dir: "Benchmark_results"
  log_file: "benchmark"
  debug_mode: true
  plot_mode: false
  save_mode: true

images_dir: "images"
labels_dir: "labels"
masks_dir: "masks"
patient_ids: ...
```

## `configs/benchmark/experiment.yaml`

Décrit l'expérience scientifique :

- format des fichiers d'entrée ;
- utilisation des unités physiques ;
- opérateur de dérivation ;
- enhancer ;
- paramètres du Hessien ;
- traitement ;
- paramètres de vesselness ;
- segmentation.

Le choix du filtre demandé sur la ligne de commande avec `--enhancer` remplace le choix de `methods.enhancer` lors de l'exécution du benchmark.

## `configs/benchmark/hessian.yaml`

Contrôle spécifiquement le benchmark Hessien :

- mode du benchmark ;
- optimisation des échelles ;
- dossier de résultats ;
- opérateurs à tester ;
- grille de recherche ;
- métriques calculées ;
- stratégie de seuillage.

La grille est filtrée automatiquement selon l'enhancer choisi dans `core/experiments/benchmarks/runner.py`.

Exemple : pour Frangi, les paramètres `alpha`, `beta` et `gamma` sont pertinents ; pour MFAT, les paramètres `mfat_tau`, `mfat_tau2`, `mfat_step_size` et `variant` sont utilisés.

---

# 6. Pipeline de traitement

## `core/processing/processor.py`

`Processor` orchestre le pipeline de traitement d'un volume.

Il coordonne :

1. préparation des données ;
2. calcul de la Hessienne ;
3. calcul de la vesselness ;
4. segmentation.

C'est le fichier à modifier si l'on veut changer **l'enchaînement général** du pipeline.

---

## `core/processing/derivator.py`

Contient la classe `Derivator`.

Responsabilité : calculer le Hessien 3D à partir d'un volume.

Les opérateurs étudiés comprennent notamment :

```text
 default
 gaussian
 farid
 cubic
 trigonometric
 catmull
 bspline
 bezier
 scharr
```

Le Hessien obtenu est ensuite transmis aux filtres de vesselness.

**Si l'on veut modifier l'implémentation d'un opérateur de dérivation, c'est ce fichier qu'il faut consulter en premier.**

---

## `core/processing/enhancer.py`

Contient la classe `Enhancer` et les méthodes de rehaussement.

Filtres disponibles dans le benchmark principal :

- Frangi ;
- Jerman ;
- MFAT.

Ce fichier reçoit la Hessienne / les valeurs propres nécessaires et produit une carte de vesselness.

**Si l'on veut modifier la formule de Frangi, Jerman ou MFAT, ajouter un filtre, c'est ce fichier qu'il faut consulter.**

---

## `core/processing/segmenter.py`

Contient la classe `Segmenter`.

Responsabilités :

- normalisation de la vesselness lorsque nécessaire ;
- recherche du seuil ;
- seuillage ;
- éventuellement double seuillage / hystérésis selon la configuration.

La fonction `_find_best_threshold_f1()` réalise la recherche du seuil optimal par F1 lorsque le seuil n'est pas fixé explicitement.

---

# 7. Chargement des données

## `core/io/loader.py`

`Loader` fournit les fonctions génériques de chargement des volumes.

Il est utilisé par le benchmark pour accéder aux images et annotations.

## `core/utils/patient_loader.py`

`PatientLoader` gère la résolution des fichiers par patient et dataset.

Il est notamment responsable de :

- trouver les images ;
- trouver les labels ;
- trouver les masques ;
- sélectionner un sous-ensemble de patients.

**Si un fichier patient n'est pas trouvé, c'est l'un des premiers fichiers à vérifier (après avoir bien vérifier la disponibilité des données).**

---

# 8. Métriques

## `core/experiments/metrics.py`

C'est le fichier central des métriques.

Il contient notamment :

### Métriques de segmentation

- `dice()`
- `mcc()`
- `roc()`
- `pr()`
- `confusion_matrix()`

### Métriques topologiques

- `cldice()`
- `largest_gt_recall()`
- `largest_component_overlap()`
- `fragmentation_ratio()`
- `skeleton_component_connectivity()`
- `connected_components_metrics()`

### Bifurcations

- `detect_bifurcations()`
- `compute_bdr()`
- `find_optimal_bifurcation_threshold()`
- `bifurcation_detection_rate()`

### Statistiques de vesselness

- `enhanced_stats()`

**Si une métrique doit être modifiée, consultée, ce fichier est le point de départ.**

---

# 9. Benchmark principal : `core/experiments/benchmarks/hessian.py`

`BenchmarkHessian` est le cœur de l'expérience de comparaison des opérateurs.

Il gère notamment :

- l'exécution patient par patient ;
- le grid search ;
- le calcul des métriques ;
- les statistiques de faux positifs ;
- les données de continuité ;
- les matrices de confusion ;
- l'analyse des bifurcations ;
- les analyses complémentaires ;
- la sauvegarde des résultats.

## Grid search

La fonction principale est :

```text
_run_grid_search()
```

Le benchmark peut chercher les meilleurs paramètres de vesselness / segmentation pour chaque cas selon la configuration.

Pour le benchmark actuel, le critère de sélection des hyperparamètres est défini dans la logique du benchmark et repose notamment sur le MCC pour la sélection des combinaisons.

## Crop utilisé pour l'optimisation

`_crop_around()` et `_get_center()` permettent de travailler sur une région centrée autour du foie / ROI lors de certaines étapes d'optimisation afin de limiter le coût du grid search.

## Confusion matrix

Les fonctions concernées sont :

```text
_log_and_store_confusion()
_create_confusion_summary_figure()
```

Les valeurs TP / FP / FN / TN sont calculées à partir de la prédiction et du GT, avec prise en compte du masque lorsque celui-ci est utilisé pour l'évaluation.

---

# 10. Où est produit le tableau benchmark ?

C'est un point important pour reproduire les tableaux du rapport.

Le tableau moyen ± écart-type est généré dans :

```text
core/experiments/analytics/runner.py
```

avec :

```text
AnalyticsRunner.get_hessian_figures()
        │
        ├── _parse_results()
        ├── _compute_statistics()
        └── _generate_mean_std_table()
```

La fonction :

```text
_generate_mean_std_table()
```

construit le texte du tableau.

Dans `get_hessian_figures()`, ce tableau est ensuite encapsulé comme une `FigureData` nommée :

```text
table_mean
```

Lors de la sauvegarde du benchmark, il est écrit dans le dossier `overview/`.

Selon le chemin de sauvegarde utilisé par le runner, on obtient donc typiquement un fichier de type :

```text
outputs/<output_dir>/<dataset>_enhancer_<enhancer>_<timestamp>/overview/table_mean.txt
```

### Tableau par patient

Il est produit par :

```text
_generate_patient_table()
```

et nommé :

```text
table_patients
```

Il se trouve également dans `overview/`.

---

# 11. Figures principales du benchmark

Les figures synthétiques principales sont générées dans :

```text
core/experiments/analytics/runner.py
```

## `overview_metrics`

Fonction :

```text
_create_overview_metrics()
```

Cette figure rassemble notamment les distributions et moyennes ± écart-type des métriques sélectionnées.

## `overview_patients`

Fonction :

```text
_create_overview_patients()
```

Cette figure compare les patients et les opérateurs à partir des résultats de Dice.

## Analyse des paramètres

Fonction :

```text
_create_overview_params()
```

Cette fonction est utilisée pour visualiser l'influence des paramètres explorés.

## Boxplots

Fonction :

```text
_create_boxplots()
```

## Radar chart

Fonction :

```text
_create_radar_chart()
```

## Distribution du Dice

Fonction :

```text
_create_dice_distribution()
```

---

# 12. `confusion_matrices.json`

Les données de confusion sont enregistrées par le benchmark dans :

```text
core/experiments/benchmarks/hessian.py
```

Les fonctions à consulter sont :

```text
_log_and_store_confusion()
_create_confusion_summary_figure()
```

Le fichier `confusion_matrices.json`, lorsqu'il est généré, appartient au dossier de sortie du benchmark.

Il permet de conserver les informations nécessaires à l'analyse des :

- vrais positifs (TP) ;
- faux positifs (FP) ;
- faux négatifs (FN) ;
- vrais négatifs (TN).

La génération de la figure de synthèse et l'enregistrement des données sont donc liés au même module de benchmark.

---

# 13. Faux positifs

Les fonctions concernées sont :

```text
core/experiments/benchmarks/hessian.py

_fp_statistics()
_save_fp_stats_incremental()
create_fp_summary_report()
```

`_fp_statistics()` calcule les statistiques sur les faux positifs.

`_save_fp_stats_incremental()` permet de sauvegarder progressivement les statistiques par patient et opérateur.

`create_fp_summary_report()` génère ensuite un rapport de synthèse.

Les résultats sont enregistrés dans le dossier de sortie de l'expérience.

---

# 14. Analyse des bifurcations

Le code est dans :

```text
core/experiments/benchmarks/hessian.py
```

avec notamment :

```text
analyze_bifurcations()
```

Les fonctions de calcul bas niveau sont dans :

```text
core/experiments/metrics.py
```

notamment :

```text
detect_bifurcations()
compute_bdr()
find_optimal_bifurcation_threshold()
bifurcation_detection_rate()
```

Cette séparation permet de distinguer :

- le calcul générique de la métrique ;
- son intégration dans l'expérience benchmark.

---

# 15. Analyse des composantes connexes
ATTENTION: uniquement après avoir lancé le benchmark, et que les résultats de ségmentation sont disponibles.

Le code de l'étude est dans :

```text
core/experiments/post_benchmark/cc_filtering/
```

Fichiers :

```text
cc_filtering_study.py
cc_filtering_analysis.py
```

Cette étude applique différents seuils de taille minimale de composante (`min_size`) et recalcule les métriques afin d'étudier l'effet de la suppression des petites composantes.

Les configurations sont :

```text
configs/post_benchmark_studies/cc_filtering/
├── ircad.yaml
├── bullitt.yaml
└── vascusynth.yaml
```

---

# 16. Tests post-benchmark
ATTENTION: uniquement après avoir lancé le benchmark, et que les résultats de ségmentation sont disponibles.

Le module principal est :

```text
core/experiments/post_benchmark/tests/post_benchmark_tests.py
```

Il regroupe plusieurs analyses statistiques effectuées à partir de résultats de benchmark déjà produits.

Les analyses comprennent notamment :

- filtrage ;
- proximité au seuil ;
- analyses de médiation ;
- distribution des seuils ;
- différences inter-opérateurs ;
- sweep de seuil.

Les configurations par dataset sont dans :

```text
configs/post_benchmark_studies/datasets/
├── ircad.yaml
├── bullitt.yaml
└── vascusynth.yaml
```

Exécution :

```bash
python3 main.py --run_post_tests --dataset ircad
```

Pour éviter le sweep de seuil :

```bash
python3 main.py --run_post_tests --dataset ircad --skip_sweep
```

---

# 17. Analyses analytiques de la Hessienne

Le dossier :

```text
core/Hessian_evaluation/
```

contient les expériences sur un fantôme synthétique `AnalyticalVessel`.

Ces expériences sont distinctes du benchmark IRCAD/Bullitt/VascuSynth principal.

Elles servent à caractériser les opérateurs indépendamment des données réelles.

---

## 17.1 `Comparaison_hessian.py`

Étudie directement l'approximation de la Hessienne.

Il contient notamment :

- `AnalyticalVessel` ;
- génération de vaisseaux synthétiques ;
- Hessienne analytique ;
- application des opérateurs numériques ;
- comparaison à la référence analytique ;
- erreur relative de Frobenius ;
- erreurs composante par composante ;
- biais d'amplitude ;
- résidu de symétrie ;
- mesure du temps d'exécution ;
- ajout de bruit ;
- génération des figures.

C'est le fichier à utiliser pour répondre à la question :

> Quel opérateur approxime le mieux la Hessienne théorique ?

---

## 17.2 `Comparaison_valeurs_propres.py`

Étudie les valeurs et vecteurs propres issus de la Hessienne.

Métriques notamment présentes :

- RMSE / erreur L2 ;
- amplification du bruit ;
- taux d'instabilité ;
- angle de rotation ;
- comparaison des valeurs propres.

C'est le fichier à consulter pour l'analyse du comportement spectral de la Hessienne.

---

## 17.3 `Comparaison_vesselness.py`

Évalue l'impact des opérateurs sur la vesselness.

Il teste notamment :

- Frangi ;
- Jerman ;
- MFAT ;
- plusieurs échelles ;
- différents opérateurs de dérivation.

Le seuil F1 optimal est recherché sur le fantôme synthétique pour l'évaluation de la vesselness.

---

## 17.4 `Comparaison_segmentation.py`

Prolonge l'étude jusqu'à la segmentation.

Le fichier relie :

```text
opérateur
   ↓
Hessienne
   ↓
vesselness
   ↓
seuillage
   ↓
metrics
```

Il permet donc de déterminer si les différences observées au niveau de la Hessienne se retrouvent au niveau de la segmentation.

---

# 18. Configuration des études analytiques

Les quatre YAML correspondants sont dans :

```text
configs/operator_study/
```

| Fichier | Étude |
|---|---|
| `hessian.yaml` | Hessienne brute |
| `eigenvalues.yaml` | valeurs/vecteurs propres |
| `vesselness.yaml` | vesselness |
| `segmentation.yaml` | segmentation |

Ils définissent notamment :

- géométrie du fantôme ;
- rayon du vaisseau ;
- sigma ;
- opérateurs ;
- cas simple / courbé / bifurcation ;
- niveau de bruit ;
- nombre de répétitions ;
- échelles ;
- filtres à tester.

---

# 19. Prétraitement des données

Le prétraitement est principalement regroupé dans :

```text
Manipulation_des_donnees/core_pre_traitement/
```

## `multi_dataset_preprocessor.py`

C'est le préprocesseur principal multi-dataset.

Il contient les classes de configuration et les pipelines :

- `IRCADPreprocessor` ;
- `BullittPreprocessor` ;
- `VascuSynthPreprocessor` ;
- `MultiDatasetPreprocessor`.

Il contient également les opérations génériques :

- resampling ;
- crop ;
- normalisation ;
- application de masque ;
- traitement streaming.

## `resampler.py`

Contient la classe `Resampler` pour le resampling des images et masques.

## `Ircad_patient/`

Contient les outils spécifiques IRCAD :

- `patient_file_resolver.py` : recherche/résolution des fichiers patients ;
- `vessel_preprocessor.py` : prétraitement spécifique des vaisseaux ;
- `Pre_processus.py` : ancien / complément de prétraitement CLI.(si on veut juste pré-traitement d'IRCAD)

## `Bullit_patient/`

Contient les outils spécifiques Bullitt :

- `bullit_file_resolver.py` ;
- `Pre_processus.py`.

## `vascusynth/`

Contient :

- `vascusynth_file_resolver.py` ;
- `tree_rasterizer.py`.

`tree_rasterizer.py` permet notamment de construire des volumes, masques GT, variantes de bruit et masques de voisinage à partir des arbres VascuSynth.

---

# 20. Configuration du prétraitement

Le fichier principal est :

```text
configs/preprocessing/dataset_config.yaml
```

Il définit les chemins et paramètres par dataset ainsi que le resampling isotrope et la normalisation.

### Attention

Le YAML fourni dans le dépôt contient actuellement des **chemins absolus spécifiques à la machine de développement** pour IRCAD et Bullitt. CHANGEZ SELON VOTRE EMPLACEMENT.

Exemple recommandé :

```yaml
input_dir: "data/ircad/..."
```

ou un chemin passé depuis une configuration locale.

---

# 21. Exploration des datasets

Le dossier :

```text
docs/
```

contient les notebooks utilisés pour l'exploration et l'audit des données.

## IRCAD

```text
docs/Exploration_données IRCAD/
├── 01_ircad_data_audit.ipynb
└── vessel_GT_liver_analysis.ipynb
```

- `01_ircad_data_audit.ipynb` : audit général des données IRCAD ;
- `vessel_GT_liver_analysis.ipynb` : analyse du GT des vaisseaux et du masque hépatique.

## Bullitt

```text
docs/Exploration_données Bullitt/
├── bullitt_exploration.ipynb
└── Avessel_GT_brain_analysis_bullitt.ipynb
```

Ces notebooks servent à l'exploration du dataset et de ses annotations.

## VascuSynth

```text
docs/Vascu-synth/
├── vasc.ipynb
└── vascusynth_DICEbenchmark_operateurs.ipynb
```

Le second notebook est orienté vers la comparaison des opérateurs sur VascuSynth.

---

# 22. Visualisation des résultats

Le dossier :

```text
visualisation_résultats/
```

contient des scripts indépendants de visualisation.

## `Visualize_skeleton_consensus.py`

Visualisation / comparaison autour des squelettes et du consensus.

## `visualiser_cv.py`

Outils de visualisation liés aux résultats de classification / confusion.

## `visualiser_volumique3D.py`

Visualisation volumique 3D des résultats.

Ces scripts ne constituent pas le cœur du benchmark : ils servent à explorer ou présenter des résultats déjà produits.

---

# 23. Configuration interne (`core/config/`)

Le dossier :

```text
core/config/
```

contient les classes qui représentent les différentes configurations du pipeline.

| Fichier | Rôle |
|---|---|
| `base.py` | classe de base des configurations |
| `builder.py` | construction des configurations à partir des YAML |
| `setup.py` | configuration générale de l'exécution |
| `engine.py` | configuration liée au moteur / exécution |
| `experiment.py` | configuration de l'expérience et du traitement |
| `benchmark.py` | configuration et structures de résultats du benchmark |
| `metrics.py` | configuration des métriques |
| `figure.py` | structure `FigureData` utilisée pour les sorties |
| `operator_study.py` | configurations des études analytiques |
| `postbench_tests.py` | configuration des tests post-benchmark |
| `cc_filtering.py` | configuration du filtrage par composantes connexes |

`builder.py` est particulièrement important : il transforme les fichiers YAML en objets de configuration utilisés par le reste du pipeline.

---

# 24. Entrées/sorties (`core/io/`)

## `loader.py`

Chargement des données.

## `saver.py`

Gestion des sorties.

Les principales méthodes sont :

```text
save_results()
save_text()
save_data()
save_plot()
save_anim()
save_figure()
save_config()
```

Le `Saver` crée automatiquement un dossier de sortie avec timestamp et, pour le benchmark, avec dataset et enhancer.

La structure générale est de la forme :

```text
outputs/
└── <output_dir>/
    └── <dataset>_enhancer_<enhancer>_<timestamp>/
        ├── results/
        ├── overview/
        └── autres sorties selon l'étude
```

Les figures sont généralement préfixées par `plot_`, les textes par `text_`, les données par `data_` et les configurations par `config_` lorsque la classe `Saver` est utilisée directement.

Le `BenchmarkRunner` possède également une logique spécifique pour sauvegarder certains objets texte dans `overview/`.

## `logger.py`

Configuration des logs de l'expérience.

---

# 25. Utilitaires (`core/utils/`)

| Fichier | Rôle |
|---|---|
| `patient_loader.py` | résolution des fichiers patients |
| `black_ridges.py` | détection / gestion de la polarité des structures |
| `searcher.py` | outils de recherche / grid search |
| `gpu.py` | détection et gestion GPU |
| `parallelizer.py` | parallélisation |
| `chunker.py` | découpage des volumes |
| `decorator.py` | décorateurs de logging / timing |
| `helpers.py` | fonctions utilitaires générales |
| `viewer.py` | outils de visualisation |
| `__init__.py` | initialisation du package |

---

# 26. Analytics (`core/experiments/analytics/`)

Le dossier `analytics` ne calcule pas le pipeline scientifique lui-même : il transforme principalement les résultats déjà obtenus en tableaux, figures et comparaisons.

## `base.py`

Classe de base pour les analytics et création de `FigureData`.

## `hessian.py`

Analytics spécifiques aux résultats Hessien :

- configurations ;
- métriques ;
- continuité patient ;
- histogrammes ;
- ROC ;
- PR ;
- vues 2D/3D.

## `runner.py`

C'est le fichier le plus important pour retrouver les **tableaux et figures synthétiques du benchmark**.

---

# 27. Relation entre résultats bruts et tableaux du rapport

Le chemin logique est :

```text
BenchmarkHessian
      │
      ├── résultats patient/opérateur
      │
      ├── métriques
      │
      ├── confusion / FP / bifurcations
      │
      ▼
BenchmarkRunner
      │
      ▼
AnalyticsRunner
      │
      ├── table_mean
      ├── table_patients
      ├── overview_metrics
      ├── overview_patients
      ├── enhanced_stats
      └── autres figures
      │
      ▼
Saver / dossier outputs/
```

C'est cette chaîne qu'il faut suivre lorsqu'un tableau ou une figure du rapport doit être reproduit.

---

# 28. Chercher un résultat, tableau, figure:

Lorsqu'une figure ou un tableau doit être retrouvé, utiliser la procédure suivante. à l'aide du rapport du stage.

### Exemple : tableau benchmark moyen ± std

```text
Rapport
  ↓
tableau benchmark
  ↓
core/experiments/analytics/runner.py
  ↓
get_hessian_figures()
  ↓
_compute_statistics()
  ↓
_generate_mean_std_table()
  ↓
table_mean
  ↓
outputs/.../overview/table_mean.txt
```

### Exemple : résultats par patient

```text
Rapport
  ↓
tableau patient
  ↓
_generate_patient_table()
  ↓
table_patients
  ↓
outputs/.../overview/table_patients.txt
```

### Exemple : matrice de confusion

```text
Benchmark
  ↓
BenchmarkHessian
  ↓
_log_and_store_confusion()
  ↓
confusion_matrices.json
  ↓
_create_confusion_summary_figure()
```

### Exemple : analyse Hessienne théorique

```text
Étude théorique
  ↓
core/Hessian_evaluation/Comparaison_hessian.py
  ↓
AnalyticalVessel
  ↓
Hessienne analytique
  ↓
application des opérateurs
  ↓
erreurs / biais / bruit / temps
  ↓
figures + tableaux de l'étude
```


---

# 29. Fichiers historiques / exploratoires

Certains fichiers ont été utilisés pendant l'exploration ou le développement et ne sont pas nécessaires pour comprendre le chemin minimal du benchmark principal.

En particulier :

```text
Manipulation_des_donnees/
core/Hessian_evaluation/
docs/
visualisation_résultats/
```

Ils sont conservés car ils documentent les différentes étapes de l'étude, mais le chemin minimal du benchmark principal reste :

```text
main.py
  ↓
configs/
  ↓
core/config/
  ↓
core/io/
core/utils/
  ↓
core/processing/
  ↓
core/experiments/benchmarks/
  ↓
core/experiments/analytics/
```

---

# 30. Dépendances

Les dépendances Python principales sont listées dans :

```text
requirements.txt
```

Un environnement GPU complémentaire est décrit dans :

```text
gpu_env.yaml
```

Les calculs peuvent utiliser CPU ou GPU selon la configuration et les capacités disponibles.

---

# 31. Reproductibilité

Pour reproduire une expérience, conserver ensemble :

1. le commit du dépôt ;
2. le YAML utilisé ;
3. le dataset ;
4. la liste des patients ;
5. l'enhancer ;
6. l'opérateur de dérivation ;
7. les paramètres du grid search ;
8. les paramètres de segmentation ;
9. le dossier de sortie produit.

Les résultats ne doivent pas être interprétés indépendamment de la configuration ayant servi à les produire.

---

# 32. Résumé : quel fichier modifier ?

```text
Je veux modifier...

→ Le calcul du Hessien
    core/processing/derivator.py

→ Un opérateur de dérivation
    core/processing/derivator.py

→ Frangi / Jerman / MFAT
    core/processing/enhancer.py

→ Le seuil / segmentation
    core/processing/segmenter.py

→ Une métrique
    core/experiments/metrics.py

→ Le benchmark
    core/experiments/benchmarks/hessian.py

→ Le lancement du benchmark
    core/experiments/benchmarks/runner.py
    main.py

→ Le tableau moyen ± std
    core/experiments/analytics/runner.py
    _generate_mean_std_table()

→ Le tableau patient
    core/experiments/analytics/runner.py
    _generate_patient_table()

→ Les figures synthétiques
    core/experiments/analytics/runner.py

→ Les matrices de confusion
    core/experiments/benchmarks/hessian.py

→ Les faux positifs
    core/experiments/benchmarks/hessian.py

→ Les bifurcations
    core/experiments/benchmarks/hessian.py
    core/experiments/metrics.py

→ Le filtrage par composantes connexes
    core/experiments/post_benchmark/cc_filtering/

→ Les tests statistiques post-benchmark
    core/experiments/post_benchmark/tests/post_benchmark_tests.py

→ L'étude théorique de la Hessienne
    core/Hessian_evaluation/Comparaison_hessian.py

→ L'étude des valeurs propres
    core/Hessian_evaluation/Comparaison_valeurs_propres.py

→ L'étude de la vesselness
    core/Hessian_evaluation/Comparaison_vesselness.py

→ L'étude de segmentation synthétique
    core/Hessian_evaluation/Comparaison_segmentation.py

→ Le prétraitement
    Manipulation_des_donnees/core_pre_traitement/multi_dataset_preprocessor.py

→ L'exploration des données
    docs/

→ La visualisation finale
    visualisation_résultats/
```

---

# 33. Chemin minimal recommandé pour comprendre le projet

Pour une personne qui découvre le dépôt, il est recommandé de lire dans cet ordre :

```text
1. README.md
      ↓
2. main.py
      ↓
3. configs/benchmark/runner.yaml
4. configs/benchmark/experiment.yaml
5. configs/benchmark/hessian.yaml
      ↓
6. core/experiments/benchmarks/runner.py
      ↓
7. core/experiments/benchmarks/hessian.py
      ↓
8. core/processing/processor.py
      ↓
9. core/processing/derivator.py
10. core/processing/enhancer.py
11. core/processing/segmenter.py
      ↓
12. core/experiments/metrics.py
      ↓
13. core/experiments/analytics/runner.py
      ↓
14. core/io/saver.py
```

Pour comprendre ensuite **pourquoi les opérateurs diffèrent**, lire :

```text
core/Hessian_evaluation/Comparaison_hessian.py
        ↓
core/Hessian_evaluation/Comparaison_valeurs_propres.py
        ↓
core/Hessian_evaluation/Comparaison_vesselness.py
        ↓
core/Hessian_evaluation/Comparaison_segmentation.py
```

Cette seconde chaîne correspond à l'étude analytique sur fantôme et doit être distinguée du benchmark sur données réelles.
