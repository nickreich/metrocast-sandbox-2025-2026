# AR6 model with theta pooling for metropolitan influenza forecasting

# To run locally without Docker

To test this out locally, run the following with `ar6_thetaP` as your working directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python main.py --today_date=2025-09-17 --short_run
```

This should result in a model output file under `../../model-output/UMass-AR6_thetaP`. There will also be some other artifacts under `AR6_thetaP/intermediate-output/output` that we can ignore for now; at some point, we would like to commit those to a different repository.

# requirements.txt and renv.lock details

`requirements.txt` and `renv.lock` were generated according to [README.md](../README.md). For `renv.lock`, we installed these specific libraries:

```bash
Rscript -e "renv::install(c('lubridate', 'readr', 'dplyr'))"
Rscript -e "renv::install('arrow')"
Rscript -e "renv::install('reichlab/zoltr')"
Rscript -e "renv::install('hubverse-org/hubData@*release')"
Rscript -e "renv::install('hubverse-org/hubVis@*release')"
Rscript -e "renv::install('hubverse-org/hubEnsembles@*release')"
Rscript -e "renv::install('reichlab/covidData')"
Rscript -e "renv::install('reichlab/idforecastutils')"  # NB: installs dev versions of above
```
