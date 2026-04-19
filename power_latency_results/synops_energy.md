# SynOps + Energy ¡ª checkpoint_best.pth

- Checkpoint: `training_outputs_RF\checkpoints\checkpoint_best.pth`
- Samples used for firing-rate avg: 225
- spike_ts: 160
- E_AC = 0.9 pJ, E_MAC = 4.6 pJ (Horowitz 2014, 45nm)

| module | type | firing rate | SynOps / sample | energy (pJ) | % |
|---|---|---:|---:|---:|---:|
| `conv1` | SNN | 0.4308 | 2,858,690 | 2,572,820.64 | 2.83% |
| `conv2` | SNN | 0.1279 | 63,380,201 | 57,042,180.86 | 62.71% |
| `conv3` | SNN | 0.0689 | 3,253,345 | 2,928,010.75 | 3.22% |
| `temp_conv1` | SNN | 0.1351 | 1,416,391 | 1,274,752.00 | 1.40% |
| `fc1` | SNN | 0.3195 | 3,350,533 | 3,015,479.30 | 3.32% |
| `cbam_conv1` | ANN | 1.0000 | 1,292,800 | 5,946,880.00 | 6.54% |
| `cbam_conv2` | ANN | 1.0000 | 1,313,920 | 6,044,032.00 | 6.64% |
| `cbam_conv3` | ANN | 1.0000 | 2,637,120 | 12,130,752.00 | 13.34% |
| `fc2` | ANN | 1.0000 | 512 | 2,355.20 | 0.00% |
| **SNN subtotal** | | | | **66,833,243.55** | 73.48% |
| **ANN subtotal** | | | | **24,124,019.20** | 26.52% |
| **CBAM total** | | | | **24,121,664.00** | 26.52% |
| **TOTAL** | | | | **90,957,262.75 pJ** = **90.9573 ¦ÌJ/sample** | 100% |
