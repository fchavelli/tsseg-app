# tsseg-app — ECML PKDD 2026 Demo Track Submission Plan

**Deadline:** 12 mars 2026 (CMT)  
**Conference:** ECML PKDD 2026, Naples, Italie (7–11 sept. 2026)  
**Format:** 4 pages Springer LNCS, single-blind  
**Vidéo:** Optionnelle (≤ 5 min, sous-titres EN), reportée au camera-ready (18 juin)

---

## 1. Positionnement

> **tsseg** est la première bibliothèque open-source unifiant ~30 algorithmes de segmentation de séries temporelles sous une API cohérente avec des métriques et datasets intégrés. Aucun concurrent n'approche cette couverture.

### Tableau comparatif

| Critère                    | tsseg      | ruptures | aeon (seg.) | tslearn |
|----------------------------|------------|----------|-------------|---------|
| Algorithmes de segmentation| ~30        | 6        | 9 (exp.)    | 0       |
| Deep Learning              | ✅ (5)    | ❌       | ❌          | ❌      |
| Méthodes bayésiennes       | ✅ (3)    | ❌       | ✅ (1)      | ❌      |
| Profil matriciel           | ✅ (3)    | ❌       | ✅ (2)      | ❌      |
| Métriques dédiées          | 10         | 3        | sklearn     | —       |
| CPD + State Detection      | ✅         | CPD seul | —           | —       |
| Démo interactive           | ✅ Gradio  | ❌       | ❌          | ❌      |
| Licence                    | AGPLv3     | BSD-2    | BSD-3       | BSD-2   |

### Arguments clés

1. **Couverture inégalée** — ~30 algorithmes couvrant statistique, bayésien, profil matriciel, clustering, deep learning
2. **API unifiée** — `fit()`/`predict()` compatible aeon `BaseSegmenter`
3. **Distinction CPD vs State Detection** — 10 métriques dédiées
4. **Démo interactive** — Gradio + HuggingFace Spaces, 5 onglets (Data, Segmentation, Comparison, Evaluation, About)
5. **Benchmarking reproductible** — Datasets intégrés + métriques unifiées

---

## 2. Structure du papier (4 pages LNCS)

| Section | Pages | Contenu |
|---------|-------|---------|
| **1. Introduction** | ½ | Problème (fragmentation des outils), gap, contribution (lib unifiée + démo) |
| **2. System Architecture** | 1 | Diagramme lib `tsseg` → app `tsseg-app`, taxonomie des algorithmes (table compacte), API `fit/predict`, métriques CPD vs State |
| **3. Demonstration** | 1 | 3 scénarios : (a) générer signal → segmenter, (b) comparer 3 algos sur MoCap, (c) évaluer avec métriques. Screenshots annotés. URL HuggingFace |
| **4. Related Systems** | ½ | Tableau comparatif ci-dessus + discussion (2 paragraphes) |
| **5. Conclusion** | ¼ | Résumé + liens GitHub / HF Spaces |
| **Références** | ½ | ~10–15 refs |

### Diagramme d'architecture

```
User ↔ Gradio UI (5 tabs) ↔ tsseg-app (params, metrics, plotting)
                                  ↕
                            tsseg library
                   ┌──────────┼──────────┐
                Algorithms  Metrics   Datasets
                (30+ det.)  (10)     (MoCap, synth)
```

---

## 3. Améliorations de l'app

### Priorité haute (avant soumission)

- [x] **4a. Guided tour** — Bouton "Load Example" (MoCap trial 0 + BinSeg pré-lancé)
- [x] **4b. Timeout exécution** — Wrapper `concurrent.futures` avec timeout de 300s
- [x] **4c. Harmonisation FR→EN** — Messages `params.py` traduits
- [x] **Auto-display dataset** — Quand les données changent, l'onglet Segmentation affiche le signal
- [x] **Métriques CPD/SD** — Bloquer les métriques incompatibles (CPD metrics seulement pour CPD algos, SD pour SD)
- [x] **Bidirectional Covering** — N'afficher que le score final (pas ground_truth_covering/prediction_covering)
- [x] **Couleurs métriques** — F1-related = bleus, Covering = verts, State metrics = palette séparée
- [x] **AMI** — Non sélectionné par défaut
- [x] **F1 margin** — Visible uniquement en contexte CPD
- [x] **Params comparison** — Affichage propre (clé=valeur, pas JSON brut)

### Priorité secondaire (camera-ready)

- [ ] Onglet "Benchmark Results" (tableau statique des résultats grid search)
- [ ] Bouton "Export as Python snippet"
- [ ] Fix concurrence fichier `/tmp` (tempfile.NamedTemporaryFile)
- [ ] Tests d'intégration Gradio
- [ ] Vidéo screencast (≤ 5 min)

---

## 4. Contexte ECML PKDD Demo Track

### Critères d'évaluation (2026)

1. **Demo Experience & Interactivité** — Ce que les participants verront/feront
2. **Innovation & État de l'art** — Nouveauté du système
3. **Valeur utilisateur & Expérience** — Utilisateurs cibles, utilité
4. **Architecture & Fonctionnement** — Composants, workflows, screenshots
5. **Comparaison & Positionnement** — Relation avec systèmes similaires
6. **Responsabilité & Fiabilité** (si applicable)

### Demos acceptés similaires (ECML PKDD 2025)

- **aeon** — Lib Python ML pour séries temporelles
- **EXTREMUM** — Outil web explications contrefactuelles sur séries temporelles
- **BellatrExplorer** — Dashboard interactif random forest
- **DetoxAI** — Python toolkit + interface

### Topics pertinents (2026)

- Pattern mining, anomaly/outlier detection, sequential & temporal mining ← **directement pertinent**
- AutoML, meta-learning
- Data quality, curation

---

## 5. Liens

- **App live:** https://huggingface.co/spaces/fchavelli/tsseg
- **tsseg lib:** https://github.com/fchavelli/tsseg
- **Documentation:** https://fchavelli.github.io/tsseg/
- **Soumission:** https://cmt3.research.microsoft.com/
