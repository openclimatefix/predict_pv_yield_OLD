# Predict PV Yield
The repo represents a significant line of experimentation on training neural networks to predict PV power output using a hybrid of numerical weather predictions (NWP) and satellite data.

Contained in the `src` directory is a library for loading PV power output, satellite images and numerical weather predictions into sensible batches. These are streamed from the OCF google cloud bucket which is not publically available. Effort has gone into making this method of streaming from cloud buckets fast (or at least useably fast).

The library itself can create sequences of PV, NWP and satellite data for the training of reccurrent networks as well as training static networks using just the last satellite image or appropriate forecast.

### Environments
There are 3 environments included for completeness to ensure everything works.

- `explicit-environment.yml` (recommended) the minimal working environment for this repo. Package versions explicitly defined.
- `environment.yml` a simplified version of the above requirements without specifying package versions.
- `overcomplete-environment.yml` the exact environment used in development, but contains many more packages than required.

### Installation
After downloading the repo and navigtaion to its root.

```
# Set up environment
conda env create -f explicit-environment.yml
conda activate predict_pv_yield
pip install -e .

# Download PV data required
mkdir data
gsutil cp -r gs://solar-pv-nowcasting-data/PV data/.
```