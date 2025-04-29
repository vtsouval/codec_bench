# 📚 ARCH Benchmark Datasets

## ✅ Integrated Datasets

	- [x] AudioMNIST — Digit classification (Speech)

### 📋 Upcoming Datasets (Planned Integration)

    - [ ] ESC-50 — Environmental sound classification
    - [ ] US8K — Urban sound classification
    - [ ] FSD50K — Human-labeled sound events
    - [ ] VIVAE — Affective non-speech vocalization classification
    - [ ] FMA-small — Music genre classification
    - [ ] MagnaTagATune — Music annotation (Multi-label)
    - [ ] IRMAS — Instrument recognition (Multi-label)
    - [ ] Medley-solos-DB — Instrument recognition (Single-label)
    - [ ] RAVDESS — Emotional speech and song classification
    - [ ] SLURP — Spoken language understanding
    - [ ] EMOVO — Emotional speech (Italian)

---


## 🗂️ Datasets Preparation

To preprocess a dataset, run the corresponding `*_gen.py` script located in the `scripts/` directory.

```bash
# Example: Preprocess the AudioMNIST dataset
python scripts/audiomnist_gen.py
```

> ⚠️ All datasets are available and downloaded from HuggingFace.

### 🛠️ Available Generators

| Dataset          | Script                   | Prepared |
|------------------|--------------------------|----------|
| AudioMNIST       | `audiomnist_gen.py`      | ✅       |
| ESC-50           | `esc50_gen.py`           | ☐        |
| US8K             | `us8k_gen.py`            | ☐        |
| FSD50K           | `fsd50k_gen.py`          | ☐        |
| VIVAE            | `vivae_gen.py`           | ☐        |
| FMA-small        | `fma_small_gen.py`       | ☐        |
| MagnaTagATune    | `magnatagatune_gen.py`   | ☐        |
| IRMAS            | `irmas_gen.py`           | ☐        |
| Medley-solos-DB  | `medley_solos_gen.py`    | ☐        |
| RAVDESS          | `ravdess_gen.py`         | ☐        |
| SLURP            | `slurp_gen.py`           | ☐        |
| EMOVO            | `emovo_gen.py`           | ☐        |

