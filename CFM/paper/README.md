# Papier : CFM Signature Lab

`CFM_Signature_Lab_paper.pdf` (34 pages) décrit le pipeline équation par équation et le compare au pipeline précédent. Le contenu va de la couche de données jusqu'à l'équilibrage de Sinkhorn, avec les résultats réels du run 1.

Pour recompiler (TeX Live, `pdflatex`, deux passes pour la table des matières) :

```bash
cd paper && pdflatex main.tex && pdflatex main.tex && cp main.pdf CFM_Signature_Lab_paper.pdf
```

- `sections/` : une section par fichier.
- `figures/` : figures du run 1 (`lab_results.zip`) et figures générées à partir de ses probabilités.
