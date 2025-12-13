# Frequency-Weighted Vocabulary Sampling

This folder contains the code used to perform **frequency-weighted data selection** for training catELMo on a reduced subset of the TCR dataset.  
The goal of this method is to prioritize training files that contain **rare 5-mer motifs**, under the hypothesis that emphasizing uncommon sequence patterns improves generalization to unseen TCRs.

## Files

### `kmer_vocab.py`
This script scans all `.txt` files in the original `train_prefix` directory and builds a **global 5-mer vocabulary**.  
It slides a window of length 5 across all TCR sequences and counts the frequency of each unique 5-mer, producing a file of the form:

<5-mer> <count>


This vocabulary is used to identify common versus rare motifs across the dataset.

### `frequency_weighted_sampling.py`
This script performs **frequency-weighted file selection** using the 5-mer vocabulary.  
Each training file is scored based on the sum of rarity-weighted unique 5-mers it contains, where rarer 5-mers contribute higher weight.  
Files are then ranked by this score, and the top fraction (e.g., 60%) is selected to form a reduced training set.

## Usage Summary
1. Run `kmer_vocab.py` on the full `train_prefix` directory to generate a 5-mer frequency file.
2. Run `frequency_weighted_sampling.py` using this vocabulary to select a subset of training files.
3. Train catELMo on the resulting reduced dataset and evaluate performance on the heldout set.

This method provides a principled, vocabulary-aware alternative to random sampling by explicitly emphasizing underrepresented sequence motifs.
