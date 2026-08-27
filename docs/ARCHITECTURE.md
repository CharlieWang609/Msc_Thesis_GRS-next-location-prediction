# Pipeline architecture

The codebase is a single Python distribution containing three import packages:

```text
mobsim -> synthetic trajectories -> mobmetric
                              `----> mobpredict -> predictions/checkpoints
                                                     |
analysis/notebooks <---------------------------------'
```

`mobsim` implements EPR, density-EPR, IPT, and DT-EPR simulation. `mobmetric`
provides behavioural validation through mobility distributions, entropy, and
daily motifs. `mobpredict` turns trajectories into model-ready sequences and
trains or evaluates LSTM, MHSA, and Mamba networks.

Configuration and runtime data are kept outside the packages. This separation
allows the installed code to remain immutable while experiments write only to
`runs/` and `outputs/`.
