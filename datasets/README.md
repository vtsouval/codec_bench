# 📚 ARCH Benchmark (extended!)

## ✅ Integrated Datasets

### 🎵 Music

- [x] FMA-small — Music genre classification
- [x] GTZAN — Music genre classification
- [x] Medley-solos-DB — Instrument recognition
- [ ] MTG-Jamendo (Genre) — Single-label genre classification
- [ ] MTG-Jamendo (Mood) — Single-label mood classification

### 🗣️ Speech

- [x] AudioMNIST — Digit classification
- [x] SLURP — Spoken language understanding (intent classification)
- [x] TIMIT — Dialect region classification
- [ ] VCTK — Speaker identification
- [x] CREMA-D — Emotional speech (English)
- [x] EMOVO — Emotional speech (Italian)
- [x] RAVDESS — Emotional speech and song classification

### 🔉 Paralinguistic / Environmental

- [x] ESC-50 — Environmental sound classification
- [x] US8K — Urban sound classification
- [x] ARCA23K-FSD — Sound event classification (subset of FSD50K)
- [x] VIVAE — Affective non-speech vocalization classification
- [x] FluSense — Illness-related sound detection (e.g., cough, sneeze)

---

## 🗂️ Datasets Preparation

To preprocess a dataset, run the corresponding `*_gen.py` script located in the `scripts/` directory.

```bash
# Example: Preprocess the AudioMNIST dataset
python scripts/audiomnist_gen.py
```
