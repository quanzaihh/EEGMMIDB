# Params / FLOPs ¡ª WrapCUBASpikingCNN (snn_n_r_dfbsa.py)

Input shape per timestep: `[B, 1, 10, 11]`  ¡¤  spike_ts = 160

FLOPs here = dense MACs (upper bound). Real SNN cost equals MACs ¡Á mean spike rate; that number is reported separately by `synops_energy.py` once a checkpoint is available.

## Per-module table

| module | params | MACs / timestep | MACs / sample | % of total |
|---|---:|---:|---:|---:|
| `conv1` | 9,856 | 41,472 | 6,635,520 | 1.15% |
| `cbam_conv1` | 526 | 8,080 | 1,292,800 | 0.22% |
| `conv2` | 84,608 | 3,096,576 | 495,452,160 | 86.09% |
| `cbam_conv2` | 2,062 | 8,212 | 1,313,920 | 0.23% |
| `avg_pool` | 0 | 0 | 0 | 0.00% |
| `conv3` | 295,680 | 294,912 | 47,185,920 | 8.20% |
| `cbam_conv3` | 8,206 | 16,482 | 2,637,120 | 0.46% |
| `temp_conv1` | 197,888 | 65,536 | 10,485,760 | 1.82% |
| `fc1` | 66,304 | 65,536 | 10,485,760 | 1.82% |
| `fc2` | 512 | 0 | 512 | 0.00% |
| neuron_state (ts_weights, cdecay/vdecay) | 160 | 0 | 0 | 0% |
| **total** | **665,802** | ¡ª | **575,489,472** | **100%** |

## CBAM aggregate (attention cost)

- CBAM params: **10,794** (1.62% of total)
- non-CBAM params: 655,008
- CBAM MACs/sample: **5,243,840** (0.91% of total)
- non-CBAM MACs/sample: 570,245,632
