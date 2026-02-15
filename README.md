# English-to-French Translation via Transformer (From Scratch)

A sequence-to-sequence (Seq2Seq) Transformer model implemented in TensorFlow/Keras to translate English sentences into French.

Unlike standard implementations that rely on pre-built abstractions, this project implements the core components of the Transformer architecture—including Multi-Head Attention, Positional Embeddings, and Causal Masking—from scratch to demonstrate a deep understanding of the "Attention Is All You Need" paper.
## Theoretical Concepts Applied

This project demonstrates the implementation of the following Deep Learning concepts:

* Self-Attention & Cross-Attention: Creating context-aware embeddings.
* Scaled Dot-Product Attention: Manually implementing the math behind the query, key, and value interactions.
* Causal Masking: Ensuring the decoder cannot "peek" at future tokens during training.
* Positional Encoding: Injecting word order information into non-recurrent architecture.
* Residual Connections & Layer Normalization: Stabilizing gradients in deep networks.

## Technical Architecture

The model follows the standard Encoder-Decoder architecture:

1. Data Pipeline:

   * Custom TextVectorization with standardization (stripping punctuation, lowercase).
   * Data splitting (Train/Val/Test) and batched prefetching using tf.data.

2. Encoder:

   * Receives English source tokens.
   * Applies Positional Embeddings.
   * Processes via Self-Attention and Feed-Forward networks.

4. Decoder:

   * Receives French target tokens (shifted right).
   * Applies Causal Masking (Masked Self-Attention).
   * Applies Cross-Attention (Querying the Encoder outputs).

5. Inference:

   * A custom decoding loop that predicts the next token iteratively until the [end] token is reached.

Code Highlight: Scaled Dot-Product Attention

I implemented the core attention mechanism manually to ensure control over the masking and scaling dimensions:

```
Python
def scaled_dot_product_attention(q, k, v, use_causal_mask=False):
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) # Matmul of Q and K
    scaled_scores = scores / tf.math.sqrt(d_k) # Scale
    if use_causal_mask:
        scaled_scores = mask_attn_weights(scaled_scores) # Apply Mask
    weights = tf.nn.softmax(scaled_scores, axis=-1) 
    output = tf.matmul(weights, v) 
    return output
```

## Performance & Results

The model was trained for 30 epochs on a dataset of ~175k sentence pairs.

  * Training Accuracy: ~97%
  * Validation Accuracy: ~93%
  * Loss: Converged to ~0.35 (Validation)

Translation Examples
|Input (English) | Model Output (French) |
|---|---|
|"Tom isn't joking." | [start] tom ne blague pas [end]|
|"We'll all be there together." | [start] nous serons tous ensemble [end]|
|"I overslept again."	| [start] jai encore trop entendu [end]|

(Note: Some translations, like the last one, show the model capturing syntax but struggling with specific idiomatic vocabulary ("trop entendu" vs "dormi trop tard"), which is expected with smaller embedding dimensions).
## Usage

Clone the repo:
```
    Bash

    git clone https://github.com/SaarangArora11/En-Fr_Transformer_fromScratch.git
```
Install dependencies:
```
    Bash

     pip install tensorflow pandas numpy matplotlib seaborn
```
Run the notebook:
    * Open en-fr-using-transformers.ipynb in Jupyter or Google Colab and execute the cells sequentially.

## Future Improvements

  > Implement Beam Search decoding for better translation quality than Greedy Search.
  > Replace the custom tokenizer with Byte-Pair Encoding (BPE) or a WordPiece tokenizer to handle out-of-vocabulary words better.
  > Scale up the embedding dimensions (currently 512) and head count for higher accuracy.

## Credits

  * Dataset: Anki/Tatoeba Project
  * Architecture Reference: Attention Is All You Need (Vaswani et al., 2017)
