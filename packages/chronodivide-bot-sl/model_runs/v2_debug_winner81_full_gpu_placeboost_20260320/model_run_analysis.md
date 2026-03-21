# Model Run Analysis: `v2_debug_winner81_full_gpu_placeboost_20260320`

**Created**: 2026-03-20 21:32 UTC (~36 min total training)
**Device**: CUDA (GPU)  |  **Epochs**: 10  |  **Best Val Loss**: 20.285 (epoch 6)

---

## Training Configuration

| Parameter | Value |
|---|---|
| Data source | `pinch_point_arabs_winner81_v2_20260319` |
| Artifact variant | v2 |
| Train shards / Val shards | 69 / 12 |
| Train samples / Val samples | 4,592 items → 36,736 action samples / 945 items → 7,560 action samples |
| Batch size | 16 |
| Learning rate | 3e-4 (cosine decay to 0) |
| Weight decay | 1e-4 |
| Grad clip norm | 1.0 |
| Window size / stride | 8 / 4 |
| LSTM core | Yes (1 layer) |
| Teacher forcing | Full |
| Family balanced sampling | Yes |
| PlaceBuilding weight boost | 1.75× |
| Action family weighting | sqrt_inverse_frequency (min 0.25, max 4.0) |

---

## Action Family Distribution & Weighting

| Family | Count | Weight | % of Data |
|---|---|---|---|
| **Order** | 27,047 | 0.189 | 73.6% |
| **Queue** | 7,227 | 0.365 | 19.7% |
| **PlaceBuilding** | 1,717 | 1.311 (×1.75 boost) | 4.7% |
| **ActivateSuperWeapon** | 415 | 1.524 | 1.1% |
| **SellObject** | 197 | 1.587 | 0.5% |
| **ToggleRepair** | 133 | 1.587 | 0.4% |
| *(Unseen family #6)* | 0 | — | — |

> [!NOTE]
> 6 of 7 action families seen. The heavy class imbalance (~74% Order) is addressed by sqrt_inverse_frequency weighting. PlaceBuilding gets an additional 1.75× multiplier.

---

## Training Curves (Loss)

| Epoch | Train Loss | Val Loss | LR |
|---|---|---|---|
| 0 | 24.578 | 22.234 | 2.93e-4 |
| 1 | 22.041 | 21.549 | 2.71e-4 |
| 2 | 20.795 | 21.030 | 2.38e-4 |
| 3 | 19.920 | 20.819 | 1.96e-4 |
| 4 | 19.218 | 20.548 | 1.50e-4 |
| 5 | 18.386 | 20.479 | 1.04e-4 |
| **6** | **17.800** | **20.285** ★ best | 6.18e-5 |
| 7 | 17.399 | 20.352 | 2.86e-5 |
| 8 | 17.183 | 20.332 | 7.34e-6 |
| 9 | 17.240 | 20.310 | 0.0 |

> [!IMPORTANT]
> **Val loss plateaued around epoch 4–6** while train loss kept dropping. The gap widened from ~2.3 at epoch 0 to ~3.1 at epoch 9, indicating **mild overfitting**. Best checkpoint was saved at epoch 6 (val loss 20.285).

---

## Final Accuracy Metrics (Val, Teacher-Forced)

| Head | Train Acc | Val Acc | Val-Free Acc | Observation |
|---|---|---|---|---|
| **Action Family** | 82.9% | 87.0% | 87.0% | ✅ Strong |
| **Specific Action Type** | 51.7% | 49.6% | 49.6% | ⚠️ Moderate |
| **Order Type** | 71.3% | 68.8% | 67.7% | ✅ Good |
| **Target Mode** | 71.1% | 69.9% | 68.9% | ✅ Good |
| **Queue Flag** | 96.2% | 95.2% | 95.1% | ✅ Excellent |
| **Queue Update Type** | 92.5% | 90.2% | 87.6% | ✅ Very Good |
| **Buildable Object** | 66.0% | 55.4% | 52.6% | ⚠️ Overfitting gap |
| **Super Weapon Type** | 80.6% | 66.7% | 66.7% | ⚠️ Overfitting (small class) |
| **Commanded Units** | 37.9% | 46.2% | 37.2% (token) / 19.7% (seq) | ⚠️ Hardest head |
| **Target Entity** | 30.1% | 30.8% | 29.6% | ⚠️ Hard |
| **Target Location** | 19.1% | 2.3% | 2.2% | 🔴 Severe overfitting |
| **Target Location 2** | 0.0% | 0.0% | 0.0% | ❌ Never activated |
| **Quantity** | 61.8% | 47.3% | 45.1% | ⚠️ Moderate gap |
| **Delay** | 20.6% | 17.4% | 16.0% | ⚠️ Hard |
| **Full Command Exact Match** | — | — | 0.10% | 🔴 Very low |

---

## Key Observations

### ✅ What Went Well
1. **Action Family classification** is strong at 87% val accuracy — the model reliably identifies *what type* of action to take
2. **Queue heads** (flag 95%, update type 90%) are near-solved
3. **Order Type** and **Target Mode** are both around 69% — reasonable
4. **GPU training** completed in ~36 minutes (10 epochs × ~3.7 min/epoch), with ~164 samples/sec throughput

### ⚠️ Concerns
1. **Target Location** has the worst overfitting: train 19.1% → val 2.3%. This is the spatial reasoning head and clearly struggles to generalize
2. **Buildable Object** shows a notable generalization gap (66% train → 53% val-free)
3. **Commanded Units** sequence exact match is only 19.7% — selecting the right *set* of units is still very hard
4. **Full Command Exact Match** at 0.1% means virtually no predictions match the full ground-truth command end-to-end
5. **Target Location 2** is completely unused (all zeros)

### 📊 Overfitting Summary
The train/val gap grew steadily. Key lossy heads where overfitting is worst:
- `targetLocation`: train loss 5.0 vs val loss 7.2 (Δ = 2.2)
- `buildableObject`: train loss 1.06 vs val loss 1.30 (Δ = 0.24)
- `quantity`: train loss 0.97 vs val loss 1.21 (Δ = 0.24)

---

## Checkpoints

| File | Size |
|---|---|
| [best.pt](file:///D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v2_debug_winner81_full_gpu_placeboost_20260320/checkpoints/best.pt) | 27.6 MB |
| [latest.pt](file:///D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v2_debug_winner81_full_gpu_placeboost_20260320/checkpoints/latest.pt) | 27.6 MB |

---

## Data Split

- **69 train shards** from diverse players (leyw7n ×5, ByE ×4, iL-nam ×4, Palacio ×3, etc.)
- **12 val shards** from separate players/games (includes dan1, Neutrino, Unstop4bl, shamou — some only in val)

> [!TIP]
> Consider potential improvements:
> - **Regularization**: Increase dropout or add more weight decay to combat target location overfitting
> - **Data augmentation**: Spatial jitter or map coordinate normalization for target location
> - **More data**: 69 train shards may be insufficient for the spatial heads
> - **Curriculum**: Train spatial heads with separate, focused loss schedules
> - **PlaceBuilding boost**: The 1.75× seems reasonable given the 4.7% representation — could experiment with higher values
