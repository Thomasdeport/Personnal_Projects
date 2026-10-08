# Papier : CFM Signature Lab

`CFM_Signature_Lab_paper.pdf` (50 pages) décrit le pipeline équation par équation et le compare au pipeline précédent. Le contenu va de la couche de données jusqu'à l'équilibrage de Sinkhorn et au vote entre voisins, avec les résultats réels du run 1 à la V3 (section 15). La section 16 couvre l'auto-apprentissage transductif (V4 : 0,6551, V5 : 0,6835) : formalisation, contrôles et diagnostics.

Pour recompiler (TeX Live, `pdflatex`, deux passes pour la table des matières) :

```bash
cd paper && pdflatex main.tex && pdflatex main.tex && cp main.pdf CFM_Signature_Lab_paper.pdf
```

- `sections/` : une section par fichier.
- `figures/` : figures du run 1 (`lab_results.zip`) et figures générées à partir de ses probabilités ; `v5_*.pdf` : figures vectorielles produites par `cfm/report.py` (run V5, plus les courbes et la calibration réassemblées avec les runs V3/V4).

Exemples chiffrés (section 7) : `python paper/examples/make_examples.py` régénère les tableaux `examples/tables/*.tex` et les figures `figures/ex_*.png` à partir du code du pipeline.
