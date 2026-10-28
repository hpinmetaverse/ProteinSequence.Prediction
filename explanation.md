# Model Design and Training Decisions

## 1. Project Objective

The project predicts protein secondary structure residue by residue from amino acid sequences. It produces two related label types:

- **Q8:** eight DSSP states: L, T, S, H, I, G, E, and B.
- **Q3:** three broader states: C (coil), H (helix), and E (strand).

Q8 is the primary prediction task because it preserves more structural detail. Q3 is useful as a simpler summary and as an auxiliary learning signal for the BiLSTM.

## 2. Dataset and Feature Decisions

The active training entry point, `scripts/train.py`, uses the CSV files:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Each row contains a `sequence` string and an `ss` Q8 label string. The sequence and label lengths are checked, and rows with mismatched lengths are discarded.

For every amino acid, the loader creates a 21-dimensional one-hot vector. The 21 positions represent the standard amino acids plus `X` for unknown residues. Unknown characters are mapped to `X` rather than causing the training process to fail.

Therefore, the models currently trained by `scripts/train.py` receive tensors shaped like:

```text
(batch_size, sequence_length, 21)
```

The repository also contains `ProteinDataset`, which can read Princeton-format NPY files and create 42-dimensional features from 21 one-hot values plus 21 PSSM values. That path is useful for future PSSM experiments, but it is not connected to the current CSV training command. The reported final metrics use the active 21-feature CSV pipeline.

## 3. Variable-Length Sequence Handling

Protein sequences have different lengths. Padding every sequence to the global maximum would waste computation, so `DynamicPaddingCollator` pads each batch only to that batch's longest sequence.

The collator returns:

- Padded input features
- Padded Q8 labels
- Padded Q3 labels
- Original sequence lengths
- A boolean mask identifying real residues

Padded labels use `-1`, which matches `CrossEntropyLoss(ignore_index=-1)`. This prevents artificial padding positions from affecting the loss or metrics.

## 4. Shared Input Representation

Both models use the same initial projection:

```text
21 input features
    -> Linear(21, 128)
    -> LayerNorm
    -> ReLU
    -> Dropout(0.3)
```

The linear layer expands the compact one-hot representation into a learned embedding space. Layer normalization stabilizes feature magnitudes, ReLU provides a nonlinear transformation, and dropout reduces overfitting.

Using the same embedding and data pipeline makes the comparison between Bi-RNN and BiLSTM more meaningful: the main difference is the recurrent architecture.

## 5. Bi-RNN Baseline

The Bi-RNN is the baseline model because it is simpler and faster than the LSTM model while still using context from both directions.

Architecture:

```text
Embedded sequence
    -> One bidirectional vanilla RNN
    -> 256 hidden units per direction
    -> 512 output features per residue
    -> Linear(512, 8)
    -> Q8 predictions
    -> Q8-to-Q3 mapping
```

A bidirectional RNN processes the sequence from both the N-terminus and the C-terminus. This is important because a residue's secondary structure can depend on neighboring residues on either side.

The Bi-RNN has no separate Q3 classifier. Its Q3 prediction is derived from the Q8 prediction using the DSSP mapping:

- L, T, S -> C
- H, I, G -> H
- E, B -> E

This design keeps the baseline small and makes Q8 the only learned output target.

### Why use it as a baseline?

A baseline provides a reference point for measuring whether the additional complexity of a BiLSTM produces a real improvement. If the enhanced model does not improve over this model, the added parameters and training time would not be justified.

## 6. BiLSTM Model

The BiLSTM is the enhanced model. It uses two bidirectional LSTM layers instead of one vanilla RNN layer.

Architecture:

```text
Embedded sequence
    -> Bidirectional LSTM layer 1
    -> Dropout(0.3)
    -> Bidirectional LSTM layer 2
    -> Dropout(0.3)
    -> Linear(512, 8) Q8 head
    -> Linear(512, 3) Q3 auxiliary head
```

Each LSTM layer has 256 hidden units per direction, producing 512 features per residue. The two layers are implemented as separate modules so dropout can be applied explicitly between them.

### Why LSTM instead of a vanilla RNN?

Vanilla RNNs can lose useful information over long sequences because gradients and hidden-state information may decay across many time steps. LSTMs use input, forget, and output gates plus a cell state to control what information is retained.

This is useful for proteins because structural context can extend over many residues. Stacking two LSTM layers also allows the first layer to learn more local patterns while the second layer combines them into higher-level sequential patterns.

### Why add a Q3 auxiliary head?

Q3 is a coarser version of Q8. Learning Q3 directly gives the shared recurrent representation an additional structural objective. The total loss for the BiLSTM is:

```text
Total loss = Q8 cross-entropy + 0.3 * Q3 cross-entropy
```

Q8 remains the primary objective, while the lower-weight Q3 term encourages useful broad structural representations without dominating Q8 learning.

## 7. Training Decisions

The training loop uses:

- Adam optimizer
- Learning rate: `0.001`
- Weight decay: `0.00001`
- Batch size: `32`
- Dropout: `0.3`
- Gradient clipping: maximum norm `1.0`
- ReduceLROnPlateau scheduler
- Masked cross-entropy loss
- Fixed random seed: `42`

Gradient clipping limits unusually large updates, which is especially useful for recurrent models. The learning-rate scheduler reduces the learning rate when validation Q3 stops improving.

## 8. Validation, Checkpointing, and Early Stopping

Validation Q3 accuracy is used as the main checkpoint-selection metric because Q3 is stable, interpretable, and directly relevant to the broader structural categories.

A checkpoint is saved only when validation Q3 improves. Each checkpoint stores:

- Model parameters
- Optimizer state
- Scheduler state
- Epoch number
- Best validation Q3
- Training history

Training can resume from that checkpoint without restarting the optimizer or scheduler. The `--resume` option also continues at the next epoch instead of repeating earlier epochs.

Early stopping prevents unnecessary computation after the validation score has plateaued. In the final BiLSTM run, training was stopped once the existing best checkpoint was considered acceptable; the retained best checkpoint was then evaluated independently on `data/test.csv`.

## 9. Evaluation Method

The evaluation script loads the checkpoint, switches the model to evaluation mode, disables gradient tracking, and runs inference over `data/test.csv`.

Metrics are calculated only over real residue positions using the collator mask. This is important because padded positions do not represent amino acids and must not inflate accuracy.

The test set is kept separate from training and validation. It is used only after model selection so that the reported test metrics provide an independent estimate of performance.

## 10. Final Results

The following values were produced by `scripts/evaluate.py` on the current `data/test.csv` split containing 256 proteins:

| Model | Test Q3 Accuracy | Test Q8 Accuracy |
|---|---:|---:|
| Bi-RNN | 68.01% | 56.78% |
| BiLSTM | 72.35% | 60.07% |

The BiLSTM improves over the Bi-RNN by:

- **4.34 percentage points on Q3**
- **3.29 percentage points on Q8**

This supports the design choice of using the stacked bidirectional LSTM as the enhanced model for this CSV-based experiment.

## 11. Interpretation and Limitations

The BiLSTM performs better than the Bi-RNN on both reported metrics, but the results should be interpreted within the current experimental setup:

- The active training pipeline uses one-hot amino-acid features only.
- The PSSM-enhanced 42-feature path is not used by these final runs.
- The reported metrics are per-residue accuracies, not per-protein accuracies.
- Q8 is harder than Q3 because it distinguishes more structural states.
- Rare classes such as beta-bridge can be difficult to predict accurately, so aggregate accuracy may hide class imbalance.

For a stronger follow-up experiment, the NPY/PSSM data path should be connected consistently to training, evaluation, and inference, and per-class precision, recall, F1, and confusion matrices should be reported alongside accuracy.
