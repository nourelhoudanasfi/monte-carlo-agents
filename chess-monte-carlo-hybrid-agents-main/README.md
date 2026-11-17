# TP Monte-Carlo aux échecs : 8 stratégies, 4 combos, 3 agents hybrides

Notebook d'apprentissage par renforcement (**First-Visit Monte Carlo**) appliqué aux échecs avec `python-chess`. Le projet compare **15 joueurs** (8 stratégies de base, 4 combos, 3 agents hybrides à 2 modèles) selon leur force, leur coût de calcul, leur rapidité à gagner et la justesse de leurs coups.

> Toute stratégie est une simple fonction `policy(board) -> move`. Elle reste donc compatible avec la simulation `pygame + pymunk` du TP d'origine : `ChessPhysicsSim.play_episode(ma_strategie)`.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `TP_MonteCarlo_Echecs_Strategies_Hybrides.ipynb` | Notebook complet (code, expériences, note de synthèse) |
| `README.md` | Ce fichier |

## Installation

Python 3.9+ recommandé.

```bash
pip install python-chess numpy pandas matplotlib tqdm jupyter
jupyter notebook TP_MonteCarlo_Echecs_Strategies_Hybrides.ipynb
```

Exécuter ensuite les cellules dans l'ordre. Les expériences utilisent `SEED = 42` pour la reproductibilité et alternent les couleurs dans chaque match.

## Plan du notebook

| Section | Contenu |
|---|---|
| **0** | Trois problèmes du TP d'origine et leurs corrections |
| **1** | Outils communs : matériel, phases de jeu, récompense, moteur `play_game`, détecteurs tactiques |
| **2** | **8 stratégies de base** |
| **3** | **4 combos** |
| **4** | **3 agents hybrides** (un modèle par cas, via un routeur) |
| **5** | Expériences (5.1 à 5.8) |
| **6** | Note de synthèse : résultats, enseignements, limites, extensions |

### Corrections apportées au TP d'origine

1. **Parties tronquées** : elles comptaient comme une défaite des Blancs. Elles sont maintenant arbitrées au matériel (écart ≥ +3 victoire, ≤ −3 défaite, sinon nulle).
2. **Épisodes animés** : ils n'apprenaient pas. L'animation est séparée de l'apprentissage.
3. **États indexés par FEN** : seulement 0,39 % des états sont revus. Le notebook utilise un **état abstrait** `(écart de matériel, phase, trait)`.

### Les stratégies

**Base (S1 à S8)** : Aléatoire, Capture gloutonne, Chasseur de mat, Glouton prudent, Minimax-2 (alpha-beta), Positionnel, Pousseur de pions, Étrangleur (minimise la mobilité adverse).

**Combos**
- **C1** Mélange ε : 70 % positionnel / 30 % aléatoire
- **C2** Changement de phase : positionnel → glouton prudent → minimax-2
- **C3** Cascade : mat en 1 > capture rentable > positionnel
- **C4** Comité : vote majoritaire de 5 stratégies

**Hybrides à 2 modèles** (classe `HybridAgent`)

| Agent | Routeur | Modèle A | Modèle B |
|---|---|---|---|
| **H1** | tactique / calme | Minimax-2 | Positionnel |
| **H2** | en avance / égal ou en retard | Étrangleur | Positionnel |
| **H3** | finale / autre phase | Minimax-2 | Glouton prudent |

### Les expériences

| # | Expérience | Mesure |
|---|---|---|
| 5.1 | Gauntlet contre l'aléatoire | Score moyen sur 30 parties |
| 5.2 | Round-robin à 15 joueurs | Matrice de scores et classement |
| 5.3 | Force vs coût | ms par coup |
| 5.4 | Rapidité | Nombre de coups pour gagner |
| 5.5 | Justesse des coups | % de coups corrects, imprécis, gaffes et mats manqués (arbitre : 2 plies + quiescence) |
| 5.6 | Ablation | Un combo bat-il ses propres composants ? |
| 5.7 | Méta-bandit ε-greedy | Apprentissage de la meilleure stratégie |
| 5.8 | MC first-visit vs TD(0) | Estimation de `V(s)` sur l'état abstrait, estimateur hybride MC/TD |

## Résultats principaux

Chiffres obtenus avec `SEED = 42`. Ils varient légèrement selon la graine et le nombre de parties.

| Stratégie | Score round-robin | ms/coup | Remarque |
|---|---|---|---|
| Minimax-2 | **0,864** | 3,5 | Meilleur joueur |
| H1 Tactique/Calme | 0,782 | 2,6 | Aussi juste que Minimax-2 pour moins de calcul |
| Positionnel | 0,768 | 0,9 | Meilleur rapport force/coût |
| C2 Phases | 0,754 | 1,2 | Victoires les plus rapides (≈ 20,5 coups) |
| C3 Cascade | 0,704 | 2,2 | Sélectionné par le bandit |
| C4 Comité | 0,457 | 8,8 | Pire rapport qualité/coût |

### Enseignements

1. **La sécurité passe avant l'agressivité** : les stratégies qui évaluent les conséquences du coup battent les stratégies à objectif unique.
2. **Voir un coup plus loin** vaut plus que n'importe quelle heuristique.
3. **Combiner n'améliore pas forcément** : les combos ne dépassent pas leur meilleur composant, et un mauvais composant contamine l'ensemble. Leur apport est surtout le coût (H1, C2).
4. **Le routeur doit séparer des cas de nature différente** : « tactique vs calme » sépare mal quand 70 à 80 % des positions sont tactiques.
5. **Le problème central du Monte-Carlo ici est l'espace d'états** : il faut abstraire pour que `V` devienne informatif.

## Limites

- Évaluation matérielle uniquement (pas de structure de pions ni de sécurité du roi).
- Arbitrage au matériel à 120 demi-coups, qui favorise les stratégies gourmandes.
- Échantillons modestes : 10 parties par paire, soit un bruit d'environ ± 0,15. Seuls les écarts supérieurs à 0,2 sont fiables.
- `V^π` dépend de la politique évaluée.

## Pistes d'extension

- Contrôle Monte-Carlo : apprendre `Q(état abstrait, stratégie)` avec ε décroissant.
- SARSA et Q-learning sur le même état abstrait.
- Approximation de fonction (régression linéaire sur matériel, mobilité, centre).
- Comité pondéré par la force mesurée ; minimax-3 avec table de transposition.
- Réinjecter les stratégies dans la simulation `pygame + pymunk`.

## Dépendances

`python-chess`, `numpy`, `pandas`, `matplotlib`, `tqdm`
