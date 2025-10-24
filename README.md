# scspecies_publication
Obtain the results of the publication "scSpecies: enhancement of network architecture alignment in comparative single-cell studies".

Computations should be performed on an Apple M2 chip with MPS GPU support.

We strongly recommend applying our package implementation [scspecies](https://scspecies.readthedocs.io/en/latest/introduction.html) to your own dataset.

## Environments

The project requires running files across five different conda environments.
These environments can be installed from its corresponding `.txt` or `.yml` file:

- `environments/scspecies_env.yml`
- `environments/metrics_env.txt`
- `environments/sysvi_scarches_scpoli_celltypeist_env.txt`
- `environments/saturn_env.txt`

## Execution Order

All files must be executed in exactly the order listed below.

1) - Download dataset adipose tissue datasets (link in File) and run `create_datasets/create_anndata_adipose.ipynb` in `scspecies_env`.
   - Download dataset glioblastoma datasets (link in File) and run `create_datasets/create_anndata_glio.ipynb` in `scspecies_env`.
   - Download dataset liver datasets (link in File) and run `create_datasets/create_anndata_liver.ipynb` in `scspecies_env`.

2) Create Muon files for cross-species alignment by running `create_datasets/get_muon_files.ipynb` in `scspecies_env`.

3) - Now the following results can be obtained: SF5 can be obtained with `Get_SF5.ipynb` in `scspecies_env`.
   - The data-level analyses in F4, SF2, SF3, SF7, SF8 can be obtained with `Get_F6_SF2_SF3_SF7_SF8.ipynb` in `scspecies_env`.
   - The aligned liver cell atlas results acroess species from F5 can be obtained with `Get_F5.ipynb` in `scspecies_env`.
   - Latent space similarity score calculation used for F3 and SF10 can be obtained with `Get_F3_SF10.ipynb` in `scspecies_env`.
   - Calculate UMAP figures used in F2 and SF1 in `Get_F2_SF1.ipynb` in `scspecies_env`.

4) Train scSpecies models with dfferent hyperparameters with `train_models/train_scspecies.ipynb` in `scspecies_env`.

5) Train competitor models. As metrics are normalized, they are important to reproduce results. 
    - Run `train_models/train_scArches_scPoli.ipynb` in `sysvi_scarches_scpoli_celltypeist_env`.
    - Run `train_models/train_sysVI_celltypeist.ipynb` in `sysvi_scarches_scpoli_celltypeist_env`.
    - Run `train_models/train_sysVI_small_dataset.ipynb` in `sysvi_scarches_scpoli_celltypeist_env`.
    - Run `train_models/train_saturn.ipynb` in `saturn_env`.

6) Calculate and visualize metrics with `Get_F4_TS3_TS4_TS5.ipynb` in `metrics_env`.
