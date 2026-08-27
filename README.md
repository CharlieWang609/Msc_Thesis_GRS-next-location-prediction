# Is Mamba a Better Choice?

[![Repository checks](https://github.com/CharlieWang609/Msc_Thesis_GRS-next-location-prediction/actions/workflows/quality.yml/badge.svg)](https://github.com/CharlieWang609/Msc_Thesis_GRS-next-location-prediction/actions/workflows/quality.yml)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Research code for the MSc thesis:

> **Is Mamba a Better Choice? A Comparison of Mamba, MHSA, and LSTM Neural
> Networks for Individual Next-Location Prediction**  
> Changling Wang · Wageningen University & Research · GIRS-2025-49 · 2025

This repository compares three sequence encoders—**LSTM**, **multi-head
self-attention (MHSA)**, and **Mamba**—for individual next-location prediction.
The models are evaluated along three dimensions:

1. predictive performance on synthetic EPR and DT-EPR trajectories;
2. robustness under causal interventions that change mobility behaviour; and
3. computational efficiency as trajectory length increases.

The study uses controlled synthetic trajectories because the underlying SBB
Green Class mobility data are privacy-sensitive and cannot be redistributed.

![Experiment pipeline](docs/figures/Data%26Methodology/flow_chart.PNG)

## Main findings

- **Predictive performance:** MHSA performed best on the less regular EPR
  benchmark, with Mamba consistently second and both outperforming LSTM. On the
  more regular DT-EPR benchmark, all three models were close; Mamba obtained the
  highest Acc@1, while MHSA led most other metrics.
- **Robustness:** models trained on DT-EPR showed almost identical degradation
  under behavioural shifts. On EPR interventions, robustness depended on the
  exploration regime: Mamba and MHSA were generally stronger at high
  exploration, while LSTM became competitive as exploration decreased.
- **Efficiency:** Mamba required the fewest FLOPs and achieved the best
  inference throughput and latency in the controlled comparison. Its advantage
  increased with sequence length. LSTM retained an advantage in training
  throughput only for very short sequences.

Overall, Mamba was not universally the most accurate model, but it offered the
strongest efficiency–scalability trade-off while remaining close to the best
model in predictive performance and robustness.


## Method

| Component | Purpose |
| --- | --- |
| EPR and DT-EPR simulators | Generate benchmark and interventional trajectories |
| Mobility metrics | Validate regularity using entropy, motifs, jump length, waiting time, and radius of gyration |
| LSTM | Recurrent baseline with linear sequence complexity |
| MHSA | Transformer-style encoder with quadratic attention complexity |
| Mamba | Selective state-space encoder with linear sequence complexity |
| Causal interventions | Change exploration through `rho`, `gamma`, and hard intervention `P` |

All prediction models share the same general structure: location and temporal
embeddings, a sequence encoder, and a fully connected next-location classifier.
This keeps the comparison focused on the encoder architecture.

## Repository structure

```text
.
|-- src/
|   |-- mobsim/       # EPR, density-EPR, IPT, and DT-EPR simulation
|   |-- mobmetric/    # mobility metrics, entropy, and daily motifs
|   `-- mobpredict/   # LSTM, MHSA, Mamba, training, and inference
|-- configs/
|   |-- simulation.yml
|   `-- prediction/   # model/loss experiment configurations
|-- analysis/         # visualisation and efficiency benchmarks
|-- notebooks/        # output-free exploratory analyses
|-- tests/             # lightweight repository checks
|-- docs/figures/      # selected thesis figures
|-- scripts/           # database setup utilities
|-- data/              # local datasets; contents ignored by Git
|-- runs/              # checkpoints and prediction logs; ignored by Git
`-- outputs/           # generated data, plots, and tables; ignored by Git
```

The local `archive/` directory contains the untouched original project folders
and thesis files. It is excluded from version control and is not part of the
public repository.

## Installation

### Conda environment

The experiments were developed for **Python 3.10**.

```bash
git clone https://github.com/CharlieWang609/Msc_Thesis_GRS-next-location-prediction.git
cd Msc_Thesis_GRS-next-location-prediction

conda env create -f environment.yml
conda activate mamba-location-thesis
```

### Pip installation

From an existing Python 3.10 environment:

```bash
pip install -e .
```

Optional dependency groups are available for specific workflows:

```bash
pip install -e ".[analysis]"          # plotting, notebooks, and FLOP analysis
pip install -e ".[database]"          # PostgreSQL-backed simulation input
pip install -e ".[mamba]"             # mamba-ssm
pip install -e ".[analysis,mamba]"    # full model-analysis workflow
```

> [!NOTE]
> `mamba-ssm` generally requires a supported Linux/CUDA build environment.
> LSTM and MHSA remain usable without it because Mamba is imported lazily.

## Data preparation

Large research datasets, trained checkpoints, and generated CSV files are
intentionally not included. The two small simulator seed inputs used by the
original project—`locs.csv` and `loc_seq.csv`—are retained so that the
simulation example remains reproducible. The expected layout is:

```text
data/
|-- simulation/
|   |-- input/          # locs.csv and loc_seq.csv
|   |-- output/
|   `-- visualization/
|-- metrics/
|   |-- input/
|   |-- output/
|   `-- output_combined/
|-- prediction/
|   |-- dtepr_inference/
|   |-- epr_inference/
|   |-- data_analysis/
|   |-- model/
|   |-- temp/
|   `-- inference/
|-- raw/              # optional GNSS preprocessing input
`-- geospatial/       # optional boundaries and basemaps
```

See [data/README.md](data/README.md) for required columns and additional notes.
Do not commit private mobility traces, credentials, model checkpoints, or large
derived datasets.

## Running the experiments

Run all commands from the repository root after installing the package.

### 1. Generate synthetic trajectories

```bash
python -m mobsim.scripts.run \
  --config configs/simulation.yml \
  --model dtepr \
  --population 8000 \
  --sequence-length 2000 \
  --seed 49 \
  --output outputs/simulation/dtepr.csv
```

Supported simulators are `epr`, `ipt`, `depr`, and `dtepr`. The YAML file
controls empirical mobility distributions and the intervention parameters
`rho`, `gamma`, and `P`.

### 2. Calculate mobility metrics

Place input CSV files in `data/metrics/input/`, then run, for example:

```bash
python -m mobmetric.scripts.run_metrics count rg dtepr
python -m mobmetric.scripts.run_entropy real dtepr
python -m mobmetric.scripts.run_motifs 0.005 relative dtepr
```

Generated plots are written to `outputs/metrics/`. The batch scripts in
`src/mobmetric/scripts/` support intervention experiments over multiple files.

### 3. Train a prediction model

Each file in `configs/prediction/` specifies a model and loss-function
combination. The repository includes LSTM, MHSA, and Mamba configurations for
cross-entropy, focal loss, weighted cross-entropy, and asymmetric loss.

```bash
python -m mobpredict.run configs/prediction/config_lstm_ce.yml
python -m mobpredict.run configs/prediction/config_mhsa_ce.yml
python -m mobpredict.run configs/prediction/config_mamba_ce.yml
```

Multiple configuration files can be passed in one invocation:

```bash
python -m mobpredict.run \
  configs/prediction/config_lstm_ce.yml \
  configs/prediction/config_mhsa_ce.yml \
  configs/prediction/config_mamba_ce.yml
```

Set `misc.training: true` for training or `false` for inference. Checkpoints,
the resolved run configuration, predictions, and performance tables are saved
below `runs/`.

### 4. Run analysis

```bash
python analysis/computational_efficiency.py
python analysis/visualize_predictions.py --help
python analysis/visualize_individuals.py --help
```

Exploratory workflows are available in `notebooks/`. Stored cell outputs have
been cleared to keep Git diffs small and reproducible.

## Experiment design

- **Benchmark datasets:** EPR and DT-EPR synthetic trajectories.
- **Split:** 60% training, 20% validation, and 20% testing.
- **Architectures:** LSTM, MHSA, and Mamba.
- **Optimiser:** AdamW with early stopping.
- **Prediction metrics:** Acc@1, Acc@5, Acc@10, MRR, and weighted F1-score.
- **Behavioural validation:** real entropy and mobility-motif proportion.
- **Interventions:** values `0.1`, `0.25`, `0.5`, `0.75`, and `0.9` for
  exploration-related parameters.
- **Efficiency metrics:** parameter count, FLOPs, throughput, and latency.
- **Thesis hardware:** NVIDIA RTX 4090 (24 GB), 16-vCPU Intel Xeon Platinum
  8352V, and 120 GB RAM.

## Reproducibility and checks

- Random seeds are exposed by the simulation and prediction entry points.
- All experiment settings are stored as versioned YAML files.
- Each prediction run saves its resolved configuration alongside its outputs.
- Data, checkpoints, generated results, credentials, and notebook checkpoints
  are excluded through `.gitignore`.
- GitHub Actions checks Python syntax, the experiment configuration matrix, and
  that notebooks contain no generated outputs.

Run the local checks with:

```bash
python -m compileall -q src analysis tests
python -m unittest discover -s tests -v
```

## Thesis

The thesis is catalogued by Wageningen University & Research:

- [Is Mamba a better choice? A comparison of Mamba, MHSA, and LSTM neural
  networks for individual next-location prediction](https://library.wur.nl/WebQuery/groenekennis/2351405)

## Citation

If you use this repository, please cite the thesis. Machine-readable citation
metadata are provided in [CITATION.cff](CITATION.cff).

```bibtex
@mastersthesis{wang2025mamba,
  author  = {Changling Wang},
  title   = {Is Mamba a Better Choice? A Comparison of Mamba, MHSA, and LSTM
             Neural Networks for Individual Next-Location Prediction},
  school  = {Wageningen University \& Research},
  year    = {2025},
  number  = {GIRS-2025-49}
}
```

## Acknowledgements

The mobility simulation, metric, and prediction foundations were adapted from
the IRMLMA projects by Ye Hong and collaborators:

- [mobility-simulation](https://github.com/irmlma/mobility-simulation)
- [mobility-metrics](https://github.com/irmlma/mobility-metrics)
- [next-location-prediction](https://github.com/irmlma/next-location-prediction)

The original Apache 2.0 notices are retained in [NOTICE](NOTICE) and
[licenses/](licenses/).

## License

Licensed under the [Apache License 2.0](LICENSE).
