# Product Requirements Document (PRD)
## Protein Secondary Structure Prediction (PSSP)
### Using Bidirectional LSTM and Bidirectional RNN

---

> **Project**: Major Project — JUET (Jaypee University of Engineering & Technology, Guna)
> **Team**: Harsh Vardhan Chauhan (231B131), Himanshu Sandip Patil (231B141), Rudransh Srivastava (231B274)
> **Supervisor**: Dr. Ajay Kumar
> **Version**: 1.0 | **Date**: October 2026

---

## 1. Product Overview

### 1.1 Problem Statement

Determining protein 3D structure experimentally (via X-ray crystallography, NMR) is time-consuming,
expensive, and technically challenging. The exponential growth of sequenced proteins in databases
(UniProt/NCBI) far outpaces experimental structure determination.

Protein secondary structure is the local spatial arrangement of the protein backbone — the critical
intermediate between amino acid sequence (primary structure) and full 3D conformation (tertiary structure).
Accurate secondary structure prediction enables:
- Rapid filtering of candidate drug targets
- Functional annotation of novel sequences
- Scaffolding for homology modelling & AlphaFold2 initialization

### 1.2 Product Vision

An end-to-end deep learning pipeline that automatically predicts protein secondary structure
(both Q8 and Q3 labels) from raw amino acid sequences, deployable as a web interface making
the tool accessible to bioinformaticians and researchers worldwide.

### 1.3 Product Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| G1 | Accurate Q8 prediction | >= 70% Q8 accuracy on CB513 |
| G2 | Accurate Q3 prediction | >= 80% Q3 accuracy on CB513 |
| G3 | Long-range context capture | Bi-LSTM outperforms Bi-RNN baseline |
| G4 | Web accessibility | Sub-5 second inference on standard hardware |
| G5 | Reproducibility | All experiments version-controlled & documented |

---

## 2. Users & Stakeholders

### 2.1 Primary Users
- **Computational Biologists**: Need rapid secondary structure prediction without expensive wet-lab experiments.
- **Drug Discovery Researchers**: Require structural predictions to identify binding sites and design therapeutics.
- **Students / Researchers**: Learning and benchmarking sequence-based structure prediction techniques.

### 2.2 Secondary Users
- **Bioinformatics Tool Developers**: May integrate the trained models as backend microservices.
- **Project Evaluators / Supervisors**: Assess research quality, reproducibility, and engineering rigor.

---

## 3. Functional Requirements

### 3.1 Core ML Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Accept raw amino acid sequences as input | P0 |
| FR-02 | Generate 42-dimensional feature vectors (21 one-hot + 21 PSSM) per residue | P0 |
| FR-03 | Predict Q8 secondary structure labels per residue | P0 |
| FR-04 | Predict Q3 secondary structure labels per residue (derived from Q8) | P0 |
| FR-05 | Handle variable-length sequences with dynamic batch padding via custom collator | P0 |
| FR-06 | Support train/validation/test split with no data leakage | P0 |
| FR-07 | Save and restore best-validation checkpoints | P0 |
| FR-08 | Implement Bi-RNN baseline model | P0 |
| FR-09 | Implement 2-layer stacked Bi-LSTM model | P0 |

### 3.2 Data Processing

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10 | Load CullPDB dataset (sequence identity < 25%, resolution < 2.5 Å) | P0 |
| FR-11 | Apply token-level sequence padding (max 700 residues) | P0 |
| FR-12 | Handle non-standard amino acid masking (NoSeq tokens) | P0 |
| FR-13 | Map target sequences to Q8 and Q3 label matrices | P0 |
| FR-14 | Evaluate on CB513 independent test set (513 non-redundant chains) | P0 |

### 3.3 Evaluation & Reporting

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-15 | Track Q3 and Q8 per-residue accuracy during training | P0 |
| FR-16 | Compute per-class precision, recall, F1 per epoch | P1 |
| FR-17 | Generate confusion matrices for Q3 and Q8 on test set | P1 |
| FR-18 | Log training curves (loss, accuracy) to disk | P0 |
| FR-19 | Compare Bi-RNN vs Bi-LSTM performance with statistical summary | P1 |

### 3.4 Web Application (Phase 5+)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-20 | Accept raw amino acid sequences via web form | P1 |
| FR-21 | Display predicted secondary structure as colored string | P1 |
| FR-22 | Provide downloadable predictions as text/JSON | P2 |

---

## 4. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Training time (full CullPDB, GPU) | < 4 hours per model |
| NFR-02 | Inference latency (single sequence, CPU) | < 5 seconds |
| NFR-03 | Code coverage (unit tests) | >= 80% for model & data modules |
| NFR-04 | Reproducibility (fixed seed) | Deterministic runs given same seed |
| NFR-05 | GPU memory footprint (batch_size=32) | < 8 GB VRAM |
| NFR-06 | Model checkpoint size | < 500 MB per model |
| NFR-07 | Documentation completeness | All public functions/classes have docstrings |

---

## 5. Constraints & Assumptions

### 5.1 Constraints
- CullPDB and CB513 datasets are used as standard benchmarks; custom datasets are out of scope for v1.
- PSSM features must be pre-computed and stored in the dataset files (PSI-BLAST not run at inference time in v1).
- Project must be completed in approximately 8 weeks following the phased schedule.

### 5.2 Assumptions
- Input sequences contain standard 20 amino acids (+masking tokens for non-standard residues).
- GPU (CUDA-compatible) is available for training; CPU inference is acceptable.
- Sequences are padded/truncated to 700 residues maximum, matching CullPDB format.

---

## 6. Out of Scope (v1)

- Real-time PSI-BLAST PSSM generation at inference time
- Tertiary or quaternary structure prediction
- Transformer/attention-based models (planned for v2)
- Multiple Sequence Alignment (MSA) as input features
- Mobile application
- User accounts / persistent prediction history

---

## 7. Success Metrics

| Metric | Baseline Bi-RNN | Enhanced Bi-LSTM |
|--------|-----------------|------------------|
| Q8 Accuracy (CB513) | >= 65% | >= 70% |
| Q3 Accuracy (CB513) | >= 78% | >= 82% |
| Training Loss Convergence | <= 30 epochs | <= 20 epochs |
| Checkpoint saved | Yes | Yes |

---

## 8. Timeline Reference

See [`PHASES.md`](./PHASES.md) for the detailed 8-week phased breakdown.
See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and model architecture details.
