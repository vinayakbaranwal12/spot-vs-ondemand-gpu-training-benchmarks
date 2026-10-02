# First-checkpoint save penalty (scratch, final rows)

Recorded: 2026-10-01T09:49:18.398652+00:00

first_checkpoint=true is the first durable save in each clean run (step 60).
Later saves are steps 80 and 100.

## 1.7B
- first median local_save_s: 15.172 s  values=[15.172212477999892]
- later median local_save_s: 10.598 s  values=[10.47969527000032, 10.716280303000076]
- first-save penalty: +4.574 s (+43.2% vs later)

## 8B
- first median local_save_s: 64.550 s  values=[64.54969534400016]
- later median local_save_s: 49.914 s  values=[49.610693402999914, 50.218206250000094]
- first-save penalty: +14.635 s (+29.3% vs later)

