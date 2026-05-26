# Synthetic evaluation

The v0.4 evaluation harness measures task loss and token accuracy for the full model and each structural ablation.

This is different from `ablate`, which only measures whether logits change for one probe sequence.

## Mixed run

```bash
python -m nctnet.cli train-lm --config configs/debug.yaml
python -m nctnet.cli eval-synthetic --checkpoint runs/debug/best.pt --seq-len 16 --size 64 --batch-size 4
```

## By-family run

```bash
python -m nctnet.cli eval-synthetic --checkpoint runs/debug/best.pt --seq-len 16 --size 64 --batch-size 4 --by-family
```

## Output

The table contains:

```text
family
ablation
task_loss
accuracy
delta_loss
delta_accuracy
tokens
```

`delta_loss` and `delta_accuracy` are measured against the full model row named `none` inside the same family.

## Families

```text
persistent_entity      expected pressure: memory, regime, residue
rule_switching         expected pressure: regime, multicard, admissibility
inversion_reciprocity  expected pressure: u/p, reversibility
contradiction          expected pressure: admissibility, residue, coherence
delayed_copy           expected pressure: memory, relations
multi_hop_relation     expected pressure: relations, memory, coherence
```

## Interpretation

A structural organ passes the first non-decoration test if disabling it changes logits.

It passes the stronger task-level test only if disabling it worsens task loss or accuracy on at least one relevant synthetic task family after matched training.

v0.4 provides the named-family harness. The next step is training schedules and reports that compare full vs ablated models after matched training budgets.
