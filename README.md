# California Housing Price Prediction

Ce projet implémente un modèle de machine learning pour prédire les prix des maisons en Californie en utilisant le dataset California Housing.

## Structure du Projet

```
california_housing/
├── data/               # Données brutes et traitées
├── models/            # Modèles entraînés
├── notebooks/         # Notebooks Jupyter
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── scripts/           # Scripts d'entraînement
├── src/              # Code source Python
│   └── california_housing/
│       ├── preprocessing/  # Outils de preprocessing
│       └── utils/         # Utilitaires
└── tests/            # Tests unitaires
```

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/yourusername/california_housing.git
cd california_housing
```

2. Installer les dépendances :
```bash
pip install -e .
```

## Utilisation

### Via les Notebooks

1. Explorer les données :
```bash
jupyter notebook notebooks/01_eda.ipynb
```

2. Entraîner et évaluer les modèles :
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

### Via le Script d'Entraînement

Pour entraîner un modèle avec les paramètres par défaut :
```bash
python scripts/train.py
```

Avec un fichier de configuration personnalisé :
```bash
python scripts/train.py --config my_config.json
```

## Configuration

Le fichier `config.json` permet de personnaliser :
- Les paramètres de preprocessing
- Les hyperparamètres du modèle
- Les paramètres de split des données

## Résultats

Le meilleur modèle obtient typiquement :
- RMSE ≈ 0.5
- R² ≈ 0.8

## Licence

MIT
