# Perceptual Anchors

This is a public repository of data and code for the paper,

> Mathias SR, Varghese L, Micheyl C, Shinn-Cunningham BG (2020). On the utility of perceptual anchors during pure-tone 
> frequency discrimination. J Acoust Soc Am 147(1): 371. PMID: [32006971](https://pubmed.ncbi.nlm.nih.gov/32006971).
> DOI: [10.1121/10.0000584](https://doi.org/10.1121/10.0000584).

To replicate the figures and tables found in this paper, simply execute `main.py`. Provided all requirements are
satisfied, all figures and tables should be generated within the appropriate directories. Due to modifications made
during the journal's editorial process, there may be some small discrepancies between the newly generated items and
those in the published paper.

Below are some details on the contents of this repo.

* `modela`
  * This directory contains the posterior samples and results related to the primary model described in the paper. If
  you want to re-fit the model to the data, perhaps after modifying something in the Python files, you need to delete
  this directory. Running `main.py` will re-generate it contents.
* `modelb`
  * Same as `modela` except for a modified version of the model where trials that included identical tones are treated
  as missing. This was done as to check that those trials did not distort the results. Since the results according to
  this model were almost identical to those from the primary model, we did not actually discuss this model in the
  manuscript except for in a small footnote.
* `data.csv` — The raw data in "one-row, one-trial" format.
* `main.py` — Script that performs everything when executed.
* `figures.py`, `models.py`, and `tables.py` — Python modules loaded by `main.py`.
* `requirements.txt` — Required Python packages.

Please note that the analysis reported in this study relied upon Python and a number of third-party Python packages
that are maintained by unpaid developers. If you are reading this many years after the study was published, it is
possible that you will encounter errors. Questions related to such errors should be directed to the relevant
development teams. For science-related questions, feel free to contact me at `samuel dot mathias at childrens dot
harvard dot edu`.



     