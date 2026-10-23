# Package architecture

```text
query_minco/
  models/                 CNN, state, 6Q decoder, 21D head; original parameter names
  trajectory/             physical decoder, MINCO adapter, explicit E1 sampler
  _vendor/minco/          unmodified upstream numerical solver + MIT notice
  inference/              explicit method/checkpoint loading and serial CPU plan
  registry.py             method ID, pinned manifest and component SHA checks
  methods/reference.json  original lineage, portable bindings and contract variants
  training/
    objective.py          original B_set forward and task/batch mean denominator
    matching/             detached rectangular Hungarian; M0/M>6 rejected
    losses/               original six-term imitation composition
    teacher/              E1 sampling helper, C1A and separate R2 search
    datasets/             explicit hash-checked cache rows and input whitelist
    recipe.py             historical AdamW and cosine LR description/functions
  evaluation/             original NumPy continuous checks and separate Goal contract
```

The inference API depends only on this package and an explicitly supplied checkpoint. Audit tools intentionally need the frozen historical workspace to establish equivalence; they are not runtime dependencies.

Training calls objective.forward -> network -> E1 postprocessor -> fixed MINCO -> physical-time teacher cost -> Hungarian -> mean(selected costs), then mean(tasks). No physics/diversity/score loss is added.

Teacher canonicalization explicitly retains the historical sampler in postprocessor_legacy.py. R2's FP64 search reference never replaces the model's FP32 inverse+bmm solver. This distinction preserves two historical numerical roles.

Historical E1 dynamic binding is tested for normal/nested/exception restoration. The new package binds E1 explicitly; it does not mutate historical module globals. The historical flight-domain variants are not unified.

No full trainer loop, deployment selector, ROS interface, or plant is provided by this migration.
