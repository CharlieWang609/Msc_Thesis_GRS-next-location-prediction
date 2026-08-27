# Local data layout

Large research data are intentionally excluded from Git. The repository retains
the two small, non-sensitive simulator seed inputs from the original project:
`simulation/input/locs.csv` and `simulation/input/loc_seq.csv`. Create or add
other files locally as needed:

```text
data/
|-- simulation/
|   |-- input/
|   |   |-- locs.csv
|   |   `-- loc_seq.csv
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
|   |   |-- intervention_result/
|   |   `-- predictions/
|   |-- model/
|   |   `-- dtepr_mhsa_demo/
|   |-- temp/
|   `-- inference/
|-- raw/              # optional GNSS preprocessing inputs
`-- geospatial/       # optional boundaries and basemaps
```

## Simulation inputs

`simulation/input/locs.csv` contains `location_id` and `geometry`; geometry
values use WKT in WGS84, for example `POINT (8.52244 47.38777)`.
`simulation/input/loc_seq.csv` contains at least `user_id` and `location_id`,
ordered by visit sequence.

## Prediction and metric inputs

Place prediction training CSV files such as `dtepr.csv` and `epr.csv` directly
under `prediction/`. Prediction CSV files use the trajectory format produced by
`mobsim`. The main fields consumed by the pipeline are `user_id`, `location_id`,
`duration`, and `geometry`; the saved dataframe index is expected to be named
`index`.

The original SBB Green Class data cannot be redistributed. Do not commit private
mobility traces, derived large CSV files, checkpoints, or credentials.

## Provenance of the included simulation inputs

The included `locs.csv` and `loc_seq.csv` files were restored from commit
`66b06271aac88475c6b91e4633dab6ac5ba5b0b8` of the upstream
[`irmlma/mobility-simulation`](https://github.com/irmlma/mobility-simulation)
repository. They are synthetic demonstration inputs rather than the private SBB
Green Class trajectories.
