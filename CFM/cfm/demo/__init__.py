"""Bibliothèque de démonstration (dossier v_demo/) : volontairement courte et commentée.

Le pipeline principal (cfm/neural, cfm/screen…) gère reprise, signatures, refit, etc. Ici, on garde
l'essentiel pour comprendre et comparer des modèles en quelques minutes chacun :

    core.py   — ouvrir le lab (cache + features + split), récupérer tableaux et séquences
    models.py — modèles simples : MLP de statistiques, CNN de tokens, GRU, petit Transformer, hybride « V2 »
    train.py  — une boucle d'entraînement lisible, la prédiction, les embeddings, les métriques
    viz.py    — un style graphique unique et les figures récurrentes
"""
