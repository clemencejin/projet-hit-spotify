# 🎵 Hit Song Prediction - Prédire les tubes musicaux à partir des données Spotify

Projet de Machine Learning (M1 IDD) visant à prédire si une chanson va devenir un *hit* à partir de ses seules caractéristiques audio.

## Contexte

Une maison de disques ou un dénicheur de talents (A&R) aimerait pouvoir repérer en amont les chansons à fort potentiel commercial, sans attendre les chiffres de streaming. Ce projet pose le problème comme une tâche de **classification binaire** : à partir des caractéristiques audio d'une chanson (énergie, dansabilité, tempo, genre...), prédire si elle fera partie des 25% les plus populaires.

## Données

- **Source :** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, via l'API Spotify)
- **Volume :** ~114 000 chansons, 20 variables (durée, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, genre, etc.)
- **Variable cible :** `is_hit`, construite en seuillant la popularité Spotify au 75e percentile → dataset déséquilibré (~75% non-hits / 25% hits), réaliste d'un vrai problème de classification.

## Méthodologie

1. **Nettoyage** : déduplication par `track_id`, traitement des valeurs manquantes.
2. **Analyse exploratoire (EDA)** : distributions, corrélations avec la cible, comparaison des features par classe, projection PCA en 2D pour visualiser la séparabilité des classes.
3. **Préparation des données** : encodage One-Hot des genres musicaux, transformations log1p sur les variables très asymétriques (instrumentalness, acousticness, liveness), séparation train/test stratifiée (80/20).
4. **Modélisation** - 6 algorithmes comparés par validation croisée (10 folds, score F1 comme métrique principale car l'accuracy est trompeuse sur un dataset déséquilibré) :
   - Régression Logistique, k-NN, Naive Bayes, Arbre de décision, Random Forest, HistGradientBoosting
5. **Gestion du déséquilibre des classes** : comparaison de trois approches (`class_weight='balanced'`, Random Undersampling, et SMOTE) pour déterminer laquelle généralise le mieux sur le test set.
6. **Évaluation finale** sur le test set (jamais utilisé avant cette étape) : courbes ROC, matrices de confusion, rapport de classification complet.

## Résultats clés

| Modèle | F1 Train | F1 CV | Accuracy Test | F1 Test |
|---|---|---|---|---|
| Logistic Regression | 0.597 | 0.596 | 0.726 | 0.594 |
| k-NN | 0.642 | 0.580 | 0.807 | 0.577 |
| Naive Bayes | 0.510 | 0.507 | 0.793 | 0.518 |
| Decision Tree | 0.513 | 0.467 | 0.577 | 0.476 |
| **Random Forest** | **0.803** | **0.628** | **0.799** | **0.636** |
| HistGradientBoosting | 0.673 | 0.628 | 0.757 | 0.624 |
| Balanced Random Forest | 0.729 | 0.622 | 0.751 | 0.623 |

- Le **Random Forest** obtient le meilleur F1 test (0.636) grâce à son mécanisme d'ensemble.
- Le **Balanced Random Forest** atteint un rappel de **0.82** sur la classe "Hit", particulièrement utile pour un cas d'usage de type "détection de talents", où l'on préfère ne manquer aucun hit potentiel.
- L'encodage One-Hot des genres musicaux s'est révélé être le levier de performance le plus important.
- La régression logistique, malgré sa simplicité, reste compétitive, preuve que les features audio linéarisent bien une partie du problème.

## Limites et perspectives

- Le score de popularité Spotify reflète une popularité récente, pas une qualité musicale intrinsèque ; le seuil au 75e percentile reste arbitraire.
- Des features additionnelles (paroles, label, date de sortie, notoriété de l'artiste) pourraient enrichir le modèle.
- Une approche en régression sur le score brut de popularité permettrait une analyse plus fine que la classification binaire.

## Stack technique

Python · pandas · scikit-learn · imbalanced-learn · matplotlib · scipy

## Contenu du dépôt

- `hit_song_prediction.ipynb` - notebook complet (EDA, modélisation, évaluation)
- `ML project.pdf` - support de présentation
- `MLproject.pdf` - rapport détaillé du projet
