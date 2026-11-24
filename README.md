# MovieLens Rating Prediction

HarvardX Data Science capstone: predicting movie ratings on the MovieLens 10M dataset.

## Files

| File | Description |
|---|---|
| `movielens.R` | Script that prepares the data, trains the models and prints the final RMSE |
| `movielens.Rmd` | Report source |
| `movielens.pdf` | Knitted report |

## Result

The final model is matrix factorization (`recosystem`). The report gives an RMSE of **0.77746** on `final_holdout_test`.

## How to run

1. Put `movielens.R` and `movielens.Rmd` in the same folder and set it as the R working directory.
2. Run `movielens.R`. It installs missing packages (`tidyverse`, `caret`, `recosystem`) and downloads the data if `ml-10M100K.zip` is not in the folder.
3. To rebuild the report, knit `movielens.Rmd`. This needs the `rmarkdown` and `gridExtra` packages and a LaTeX installation such as TinyTeX. Knitting reruns the whole script, so it is slow.

Results can differ in the last decimals between runs because the model trains on several processor threads.

If the automatic download fails, download `ml-10m.zip` from <https://grouplens.org/datasets/movielens/10m/> and save it in the same folder as `ml-10M100K.zip`.
