# Reinforcement-learning configuration layout

RL configuration is split into reusable components and runnable experiments:

```text
rl/
├── components/
│   ├── runtime/       # base checkpoint, outputs, seeds, and replay settings
│   ├── algorithms/    # PPO or GRPO selection
│   ├── methods/       # plain, PIRL, and PIPO parameters
│   ├── optimization/  # policy optimization parameters
│   ├── validation/    # dual-baseline evaluation parameters
│   └── rewards/
│       ├── contracts/   # target definitions and safety constraints
│       └── registries/  # predictor checkpoints and audit metadata
└── experiments/       # runnable task and schedule configurations
```

An experiment declares relative component files through `includes`. Includes
are merged in listed order; fields in the experiment file override all included
values. For example:

```yaml
includes:
  - ../components/runtime/base.yaml
  - ../components/algorithms/grpo.yaml
  - ../components/methods/pipo.yaml
  - ../components/optimization/policy.yaml
  - ../components/validation/dual_baseline.yaml

property: fe
updates: 200
group_size: 16
```

Run the experiment by passing the experiment file itself:

```bash
python -m scripts.cli.training.train_crystal_rl \
  --train-config conf/rl/experiments/grpo_fe_seed42_pipo_probe32.yaml
```

All RL command-line tools load experiments through
`cgdit.rl.config.load_rl_config`; direct `OmegaConf.load` is appropriate only
for leaf components such as reward contracts or predictor registries.
