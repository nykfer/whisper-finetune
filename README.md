# Whisper fine-tuning: training signals and UMAP analysis

Notes on what to monitor while training a model (with a focus on Whisper / ASR) and how UMAP embedding maps are used
to find data and model problems before and after training.

## Part 1: What to log during training

Loss tells you *if* training goes well; these signals tell you *why* it doesn't.

### 1. Gradient norm
- **What:** one number per step, how strongly the model wants to change its weights.
- **Why:** called the single most useful signal for spotting broken training.
  - Sudden spikes → exploding gradients (NaNs are usually close behind).
  - Slowly fading to ~0 → vanishing gradients, the model stopped learning.
- **Tip:** compute it after `loss.backward()` and before `optimizer.step()`. If you clip gradients, log the value
  *before* clipping, otherwise the spikes are hidden.

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # returns the pre-clip norm
print(f"step {step} | loss {loss:.4f} | grad_norm {total_norm:.3f} | lr {lr:.2e}")
```

### 2. Update-to-weight ratio (per layer)
- **What:** how big each update is compared with the weight itself.
- **Rule of thumb (DL4J docs):** about **1e-3** (log10 ≈ -3) is healthy.
  - Much bigger than 1e-2 → learning rate too high, the model is shaken too hard.
  - Smaller than 1e-4 → learning rate too low, the layer barely moves.
- These thresholds come from one framework's documentation: starting points, not laws.

### 3. Activation and gradient spread per layer
- Log the standard deviation of each layer's outputs. Values drifting very large or collapsing toward 0 usually point
  to initialization or input-normalization problems.

### 4. Learning rate
- Always print it. Many "mysterious" problems are a warmup or scheduler bug.

### 5. Train loss vs. validation metric
- Train loss falls while validation gets worse → overfitting.
- Flat, high loss is not always instability: it can be a learning rate that is too low or a data bug.
- Look at trends over many steps, not single noisy values.

### Whisper-specific signals

| What to log | Why |
|---|---|
| **WER on a held-out set** | More meaningful than loss. Pick the checkpoint by best WER, not lowest loss. |
| **A few decoded sample transcripts** at each eval | Shows repetition loops ("the the the…"), wrong language or missing punctuation long before WER explains it. |
| **Hallucination rate on non-speech audio** (silence, noise, music) | WER cannot see invented text on silence. Calm-Whisper tracks this separately. |
| **Output length / reference length ratio** | < 1 means dropped words; > 1 means hallucinations or repeats. |
| **Per-subset WER** (speaker, accent, noise level, clip length) | One average can hide a group that is getting worse. |

Practical Whisper findings:
- **LoRA alpha:** one Swiss German study found the common `alpha = 2 × rank` setting caused hallucinations; lowering
  alpha/r to about 0.2 fixed it. One team's result, but cheap to test.
- **Silence mismatch:** if training clips have almost no leading silence, the model can hallucinate on real audio that
  starts with silence. Trimming initial silence to about 0.3 s removed most of those hallucinations in that study.
- **VAD:** running voice activity detection before transcription (as WhisperX does) reduces hallucinations and repeats.
- **Fine-tuning only hallucination-prone attention heads** (Calm-Whisper) cut non-speech hallucinations by >80% with
  <0.1% WER degradation on LibriSpeech.
- **Small datasets overfit:** in one reported case (~200 h of clips) gains only came after adding more data.

## Part 2: UMAP for finding problems

**Idea:** turn every sample (audio clip, sentence, image) into an embedding vector with a pretrained model. UMAP
squashes the vectors into 2D so they can be viewed as dots on a map; similar samples land near each other. Color the
dots by something (label, WER, source, speaker) and look for odd patterns.

### What it can show
- **Isolated dots / small islands:** outliers — corrupted audio, silence, wrong language, clipped recordings.
- **Tight clumps:** near-duplicates, which inflate scores if they end up in both train and test.
- **Dots whose color doesn't match their neighbors:** possible label errors.
- **Train and test forming separate clouds:** domain shift.
- **High-WER samples concentrated in one region (after training):** a failure mode the model hasn't learned (an accent,
  noisy phone audio, a specific microphone).

### Audio
- A DCASE anomaly-detection paper used UMAP to inspect machine-sound data. Key warning: **UMAP can create fake
  structure.** Compute outlier scores (kNN distance, LOF) on the original high-dimensional embeddings and use UMAP
  only to view them.
- Renumics Spotlight does this interactively: load a dataframe, get a UMAP "similarity map", click a dot to listen. It
  can color dots by model uncertainty to find label errors.
- Cleanlab has a tutorial for finding mislabeled audio using embeddings plus a simple classifier (not UMAP itself, but
  a good companion).
- Overlapping clusters are not automatically a problem — background noise naturally blurs them.

### Whisper
- A recent paper looked at Whisper-small encoder embeddings layer by layer (with t-SNE, a UMAP cousin): clusters by
  the speaker's native language get clearer in deeper layers. The encoder encodes accent, so a map can show which
  accents a training set covers or misses.
- **Caveat:** speaker identity mixes heavily with accent/dialect. A cluster may be "one speaker" rather than "one
  accent". Per-utterance normalization can reduce the speaker effect.
- Suggested workflow (a combination of the above, not a documented method):
  1. Run the dataset through the Whisper encoder and mean-pool over time → one vector per clip.
  2. **Before training** (base model): color by data source, length, speaker → find outliers, duplicates,
     train/test mismatch.
  3. **After training** (fine-tuned model): color by per-clip WER and hallucination flag → find regions where the
     model fails, then listen to those clips.
  4. Compare before/after maps. Collapsing or blurring clusters may indicate the encoder is forgetting general
     knowledge (catastrophic forgetting).

### Text
- Same approach with sentence embeddings (e.g. transcripts).
- A 2026 annotation-workshop paper used UMAP + silhouette scores to flag annotation errors in NER data; all five
  top-flagged outliers were real errors.
- Embedding reference transcripts can reveal odd text: wrong language, garbage, inconsistent formats (digits vs
  spelled-out numbers), subtitle-style text that doesn't match the audio.

### Main caveat
A UMAP plot is a rough map, not proof. Distances between far-apart clusters mean little. Always open the actual samples
behind a surprising cluster before concluding anything.

## Sources

- [Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks](https://arxiv.org/pdf/2102.06604)
- [ApX – Monitoring training metrics](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities/monitoring-training-metrics)
- [LightlyTrain – Model Instability Debugging](https://docs.lightly.ai/train/stable/debugging/model_instability.html)
- [DL4J – Visualization / update ratio](https://deeplearning4j.konduit.ai/deeplearning4j/multilayernetwork/visualization)
- [Towards Data Science – BERT training monitoring](https://towardsdatascience.com/?p=172382)
- [Calm-Whisper: Reduce Whisper Hallucination on Non-Speech](https://arxiv.org/pdf/2505.12969)
- [Subtitle-Aligned Fine-Tuning of Whisper for Swiss German ASR](https://arxiv.org/pdf/2606.07608)
- [HF discussion – hallucination after fine-tuning Whisper-small](https://huggingface.co/spaces/openai/whisper/discussions/100)
- [Diabolocom – Fine-tuning ASR, focus on Whisper](https://www.diabolocom.com/research/fine-tuning-asr-focus-on-whisper)
- [f22labs – How to fine-tune Whisper small](https://www.f22labs.com/blogs/how-to-fine-tune-whisper-small-for-better-speech-recognition/)
- [Using UMAP to Inspect Audio Data for Unsupervised Anomaly Detection (DCASE)](https://arxiv.org/pdf/2107.10880)
- [Parsing Birdsong with Deep Audio Embeddings](https://arxiv.org/pdf/2108.09203)
- [Cleanlab – Finding label issues in audio classification datasets](https://cleanlab.ai/blog/label-errors-audio-datasets/)
- [Renumics – How to quickly find and correct label errors](https://renumics.com/blog/finding-label-errors/)
- [Renumics Spotlight Playbook](https://renumics.com/docs/playbook)
- [Linguistic Distance Segregates Latent Representations in ASR Systems](https://arxiv.org/pdf/2608.30853)
- [Analyzing Speaker Information in Self-Supervised Models](https://arxiv.org/pdf/2108.00917)
- [Accent/dialect identification with multi-embedding models](https://arxiv.org/pdf/2310.11004)
- [Theodo – Dimensionality reduction for exploration and curation of datasets](https://blog.theodo.com/2024/03/dimensionality-reduction-for-exploration-and-curation-of-datasets)
- [Text dataset viewers need better embedding context](https://nhimg.org/articles/canica-shows-why-text-dataset-viewers-need-better-embedding-context/)
