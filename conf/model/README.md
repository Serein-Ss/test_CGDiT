# Model configuration layout

Model configuration is divided by responsibility:

```text
model/
├── diffusion/
│   ├── standard.yaml    # standard diffusion-model parameters
│   ├── multimodal.yaml  # multimodal diffusion-model parameters
│   ├── with_type.yaml   # legacy atom-type diffusion variant
│   ├── beta/            # continuous-time beta schedules
│   ├── sigma/           # wrapped-coordinate sigma schedules
│   ├── d3pm/            # discrete atom-type diffusion schedules
│   └── decoder/         # denoising network architectures
├── property_predictors/
│   ├── backbone/             # reusable network architecture parameters
│   │   ├── diffusion_cspnet.yaml
│   │   └── m3gnet.yaml
│   ├── diffusion_cspnet/     # tasks implemented with the CSPNet backbone
│   │   ├── tc_regressor.yaml
│   │   └── tc_classifier.yaml
│   └── m3gnet/               # tasks implemented with the M3GNet backbone
│       ├── regression.yaml
│       └── tc_regressor.yaml
└── experiments/             # complete Hydra-selectable model experiments
```

Experiment files compose reusable components with Hydra defaults. They should
contain only experiment-specific overrides, such as conditioning properties,
normalization statistics, multimodal curriculum changes, or model variants.

Select an experiment with its path relative to the `model` config group:

```bash
python -m cgdit.run \
  data=mp_20 \
  model=experiments/exp_mp20_fe \
  expname=mp20_fe
```

Select a property predictor in the same way:

```bash
python -m cgdit.run \
  data=magndata \
  model=property_predictors/diffusion_cspnet/tc_regressor \
  expname=magndata_tc
```

Component files are not complete experiments and should not be selected as the
top-level `model` option. Add reusable diffusion parameters or architectures
under `diffusion/`; add a runnable model combination under `experiments/`.
