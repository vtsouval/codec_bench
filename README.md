# 🎧 Neural Audio Codecs for Audio Classification

## Abstract

We propose a novel framework for **audio classification using large language models (LLMs)** in conjunction with **neural audio codecs**. Instead of processing raw waveforms or spectrograms, our pipeline encodes audio using a **self-supervised neural audio codec (XCodec2)** and feeds the resulting semantic token stream into a **pre-trained multilingual language model (Llasa-1B)**.

This approach treats audio classification as a **sequence-to-sequence instruction-following task**, where class labels are embedded into natural language prompts. It leverages the multimodal token space of Llasa-1B, enabling classification across diverse domains without architectural changes.

We benchmark our system on the **ARCH benchmark**, which comprises a wide variety of datasets across **sound event**, **music**, and **speech** classification tasks.

---

## 🔧 Components

- **Audio Codec**: `XCodec2` (neural codec pretrained on speech/audio)
- **LLM Backbone**: `Llasa-1B-Multilingual`
- **Prompt Style**: `<|TEXT_UNDERSTANDING_START|>Classify the audio in the following segment.<|TEXT_UNDERSTANDING_END|> → Label`

---

## 🎯 Task Types

- Single-label classification (e.g., digit recognition, emotion detection)
- Multi-label classification (e.g., instrument tagging)
- Categories: **Sound Events**, **Music**, **Speech**

---

## 📚 ARCH Benchmark Datasets

| Dataset           | Type         | Domain       | Integrated |
|------------------|--------------|--------------|------------|
| ESC-50           | Single-label | Sound events | ☐          |
| US8K             | Single-label | Sound events | ☐          |
| FSD50K           | Single-label | Sound events | ☐          |
| VIVAE            | Single-label | Sound events | ☐          |
| FMA-small        | Single-label | Music        | ☐          |
| Medley-solos-DB  | Single-label | Music        | ☐          |
| RAVDESS          | Single-label | Speech       | ☐          |
| AudioMNIST       | Single-label | Speech       | ✅         |
| SLURP            | Single-label | Speech       | ☐          |
| EMOVO            | Single-label | Speech       | ☐          |

---