# Implementing a basic version of the original encoder/decoder transformer from scratch

Our goal is to implement the following architecture from the original transformer:

![Original Transformer Architecture](static/the-annotated-transformer.png)

## Core symbols

- B: batch size
- S: source sequence length
- T: target sequence length
- D: model width, usually `d_model`. For the original paper = 512
- H: number of heads. For the original paper = 8
- `d_k`: key/query width per head. For the original paper = 64
- `d_v`: value width per head. For the original paper = 64
- `d_ff`: feedforward network depth. For the original paper = 2048

The classic setup enforces:

- `D % H == 0`
- `d_k = d_v = D // H`

## Tensor convention

Let's do `batch_first=True`. This gives:

- token embeddings: `(B, L, D)`
- positional encodings: `(B, L, D)`

## The full encoder + decoder system

The full transformer system, end-to-end, flows like this:

Input $\rightarrow$ Encoder $\rightarrow$ Context $\rightarrow$ Decoder $\rightarrow$ Output

At a high-level, we can imagine encoders vs. decoders as something like:

- Encoders = understanding
- Decoders = generation

When used together, encoders tell us "what does this input mean?", and then that representation of "understanding" is used by the decoder to tell us "given that meaning, what should we produce next?"

## Transformer architecture input/output

$$
\text{Input} \rightarrow [\text{Encoder} \rightarrow \text{Context} \rightarrow \text{Decoder}] \rightarrow \text{Output}
$$

Let's for now treat the transformer architecture as a black box, and examine what goes into it.

$$
\text{Input} \rightarrow \text{[Black box]} \rightarrow \text{Output}
$$

### Generating the input

Let's say that you have the input "I went to the bank to deposit money":

```python
string = "I went to the bank to deposit money"
```

#### Tokenization

Here, let's say that we have `S=8` tokens from this string. Let's assume a big oversimplification of the tokenization and just make each word its own token.

So, this gives us:

- `B=1` because we have 1 sentence.
- `S=8` because we have 8 source tokens.

```python
tokenized_string: list[str] = ["I", "went", "to", "the", "bank", "to", "deposit", "money"]
```

Let's imagine that each token is mapped to an integer ID:

```python
tok_to_int_map = {} # some existing map
token_ids = [tok_to_int_map[token] for token in tokenized_string]
```

#### Creating embeddings

Then let's imagine that we have a lookup map that maps ids to an embedding table (this basically is what happens during embedding).

```python
tok_id_to_embedding: dict[int, list[float]] = {} # assume this exists. list[float] has len=512

# we represent this as a list of length 1 as well, to match the (B, S, d_model) format.
# we do this because in practice, we'd take a bunch of input strings in parallel and
# create the token embeddings. Here, we just 1 string.
token_embeddings = [[tok_id_to_embedding[token_id] for token_id in token_ids]] # shape: (B, S, d_model) = (1, 8, 512)
```

This gives us `D=512` as the size of the hidden representation for the model (copying what was in the original transformers paper, which also used `d_model=512`). Each token vector has 512 features/parts. The input token embedding is 512-dimensional. The positional embedding is 512-dimensional. The hidden state throughout the encoder/decoder stack is also 512-dimensional. Keeping this consistent dimensionality is very important.

#### Positional encodings

We then add positional encodings. This lets the model know where each token occurs in the sequence.

Self-attention, by itself, doesn't know token order, so we inject position information (i.e., "this token was token 1 in the sequence") into each token vector before it enters the encoder.

For example, if we have the following token sequences:

- `["dog", "bit", "cat"]`
- `["cat", "bit", "dog"]`

These are the same tokens in different order. Without positional information, the model would have a hard time distinguishing between them just based on the tokens. As it turns out, separating the semantic meaning of a token and the positional information of a token helps improve training. This dual-encoding lets us split the task of understanding into two parts:

We need a way to represent the position using the same `d_model`-dimensional size embedding (here, called "encoding") as the token embeddings. Each position gets its own vector and that vector is added to the token embedding.

$$\text{encoder_input} = \text{token_embeddings} + \text{positional_encodings}$$

We don't use the literal index position like `0, 1, 2, ...` because a single integer per token isn't in the same vector space as the embedding. A single scalar index also doesn't give the model a representation of position that is encoded across many dimensions. We need the position information to interact with token meaning in a flexible way, so we need a representation that can be in the same `d_model`-dimensional space. The original transformer uses sinusoidal positional encodings because they were found to work well in practice and give a simple fixed way to represent position at every token slot.

Let's create a simplified version of a positional encoding function, following a version of the sinusoidal positional encoding that was used in the original transformers paper.

```python
import math

def _create_single_positional_encoding(position: int, d_model: int) -> list[float]:
    """Get the positional encoding for a single position.
    
    For a given position, its positional encoding must be `d_model`-dimensional,
    so that we can add it to the `d_model`-dimensional token embeddings.
    """
    for i in range(d_model):
        exponent = (2 * (i // 2)) / d_model
        angle = position / (10000 ** exponent)
        if i % 2 == 0:
            return math.sin(angle)
        else:
            return math.cos(angle)

def get_positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    positional_encodings: list[list[float]] = []
    for position in range(seq_len):
        # for a single position, get its corresponding appropriate
        # positional encoding. For example, for position = 1, get
        # the sinusoidal encoding for it.
        single_position_encoding: list[float] = (
            _create_single_positional_encoding(
                position=position,
                d_model=d_model
            )
        )
        positional_encodings.append(single_position_encoding)
    return positional_encodings

D = 8
tokens = ["I", "went", "to", "the", "bank", "to", "deposit", "money"]
S = len(tokens)

positional_encodings = get_positional_encoding(S, D)
print(len(positional_encodings))        # 8 positions
print(len(positional_encodings[0]))     # 8 features per position
```

#### Creating the input to the encoder

Once we have this, our encoder input becomes the addition of the token embeddings with the positional encodings:

```python
# example token embeddings
token_embeddings = [[
    [0.20, -0.10, 0.05, 0.30, 0.11, -0.07, 0.40, 0.02, ...],  # "I" (imagine d_model=512)
    [0.18,  0.09, 0.21, 0.14, 0.03,  0.25, 0.08, 0.17, ...],  # "went"
    [0.01,  0.12, 0.33, 0.07, 0.19, -0.04, 0.22, 0.31, ...],  # "to"
    [0.15, -0.03, 0.11, 0.27, 0.05,  0.18, 0.09, 0.24, ...],  # "the"
    [0.41,  0.06, 0.17, 0.12, 0.28,  0.03, 0.14, 0.10, ...],  # "bank"
    [0.01,  0.12, 0.33, 0.07, 0.19, -0.04, 0.22, 0.31, ...],  # "to"
    [0.29,  0.08, 0.13, 0.35, 0.16,  0.05, 0.27, 0.06, ...],  # "deposit"
    [0.32,  0.11, 0.24, 0.18, 0.21,  0.09, 0.30, 0.04, ...],  # "money"
]] # Shape = (B, S, D) = (1, 8, 512)

D = 512

positional_encodings: list[list[list[float]]] = [ # Shape = (B, S, D) = (1, 8, 512)
    get_positional_encoding(len(tokenized_input), D)
    for tokenized_input in token_embeddings
]

encoder_inputs: list[list[list[float]]] = [] # Shape = (B, S, D) = (1, 8, 512)

# iterate through each sample input in the batch
for sample_input_embedding, positional_encoding in zip(token_embeddings, positional_encodings):
    # sample_input = (S, D) = (8, 512). This is the tokenized embeddings of the 1st input of the batch.
    # positional_encoding = (S, D) = (8, 512). This is the positional encoding of the 1st input of the batch.
    sample_encoder_input = sample_input_embedding + positional_encoding # shape = (S, D) = (8, 512)
    encoder_inputs.append(sample_encoder_input) # shape = (B, S, D) = (1, 8, 512). We append a series of lists (S, D) into another list, B times.

return encoder_inputs
```

We can put this into a single function:

```python
d_model = 512
tok_to_int_map = {} # some existing map. Size = d_vocab. Maps tokens to ints.
tok_id_to_embedding: dict[int, list[float]] = {} # assume this exists. list[float] has len=512

texts: list[str] = [ # Shape = (B, S) = (2, 8)
    "I went to the bank to deposit money",
    "You ran to the beach to exercise more"
]
def get_encoder_inputs(batch_text_inputs: list[str]) -> list[list[float]]: # shape = (B, S, D) = (2, 8, 512)
    """Given a list of `B` text_inputs, get the inputs to the encoder."""
    tokenized_strings: list[list[str]] = [ # shape = (B, S) = (2, 8)
        tokenize_string(text_string) # shape = (S). For simplicity, let's assume all strings are the same length. In practice, we'd guarantee this with padding.
        for text_string in batch_text_inputs
    ]
    # get the token IDs for each string. Shape = (B, S) = (2, 8)
    # looks like: [[2, 3, 4, 10, 38, ...], [3, 40, 1, 991, ...]]
    batch_token_ids = [] # Shape = (B, S) = (2, 8)
    for single_sample_tokenized_string in tokenized_strings: # loops `B` times, one for each word.
        token_ids = [tok_to_int_map[token] for token in single_sample_tokenized_string]
        batch_token_ids.append(token_ids) # Shape = (S) = (8)

    ### CREATING THE EMBEDDINGS ###

    # Convert each token in (B, S) with a d_model-dimensional embedding. This
    # leads to (B, S, D) output.
    batch_token_embeddings = []
    for single_sample_token_ids in batch_token_ids:
        # Shape = (S, D) = (8, 512). Maps each of the tokens in (S) to a D-dimensional embedding.
        # Input = (S) => Output = (S, D)
        token_embeddings = [[tok_id_to_embedding[token_id] for token_id in single_sample_token_ids]]
        batch_token_embeddings.append(token_embeddings) # Appends (S, D) embeddings `B` times => (B, S, D)

    ### CREATING THE POSITIONAL ENCODINGS ###
    batch_positional_encodings = [] # same (B, S, D) shape as `batch_token_embeddings`, since these have to be exactly added.
    for single_sample_token_ids in batch_token_ids: # doesn't really matter which list we loop from, as long as we get an S-length object.
        total_tokens = len(single_sample_token_ids) # S
        positional_encodings = get_positional_encoding(S=total_tokens, D=d_model) # output = (S, D) = (8, 512)
        batch_positional_encodings.append(positional_encodings) # Appends (S, D) embeddings `B` times => (B, S, D)
    
    batch_encoder_inputs: list[list[list[float]]] = [] # Shape = (B, S, D) = (2, 8, 512)

    # Loops through `B` times, and each of `single_sample_embedding` and `single_sample_positional_encoding` is (S, D) => (8, 512)
    for (single_sample_embedding, single_sample_positional_encoding) in zip(batch_token_embeddings, batch_positional_encodings):
        single_sample_encoder_input: list[list[float]] = single_sample_embedding + single_sample_positional_encoding # Output = (S, D)
        batch_encoder_inputs.append(single_sample_encoder_input) # Appends (S, D) embeddings `B` times => (B, S, D)
    
    return batch_encoder_inputs

encoder_inputs: list[list[list[float]]] = get_encoder_inputs(texts)

print(len(encoder_inputs)) # First dimension, B = 2
print(len(encoder_inputs[0])) # Second dimension, S = 8
print(len(encoder_inputs[0][0])) # Third dimension, D = 512
```

### Passing the input through the encoder

(let's now pass it into the encoder. Let's assume the encoder is a black box...)

(compare the representations before/after encoder. Should be the same, but now each embedding representation has been modified).

### What goes into the decoder?

#### Why is the transformed representation useful to the decoder?

### What comes out of the decoder?

### Zooming out: how has the input changed when it comes out?

## Encoder layer deep dive

### What is the purpose of the encoder?

### Encoder block

#### Input to one encoder layer

- `x = (B, S, D)`

#### Learned projections

Input -> Q, K, V -> Output (O)

- `W_Q: D -> H * d_k`
- `W_K: D -> H * d_k`
- `W_V: D -> H * d_v`
- `W_O: H * d_v -> D`

#### After projection and reshape

- `Q: (B, H, S, d_k)`
- `K: (B, H, S, d_k)`
- `V: (B, H, S, d_v)`

#### Attention score tensor

Multiplying the queries and keys.

- `scores = Q @ K^T`
- Shape: (B, H, S, S)

##### What do we do this for?

(TODO: add explanation)

##### Scaling, masking (if necessary), and softmax

Once we have the attention scores, we do scaling, masking (if necessary) and softmax. The resulting attention weights have the same shape, `(B, H, S, S)`.

#### Weighted sum with `V`

Context per head: `(B, H, S, d_v)`

#### After concatenating heads

- `Shape: (B, S, H * d_v)`

#### After output projection

- `Shape: (B, S, D)`

#### Residual path

After the multi-head attention mechanism, we have a residual path.

- `Input (residual): (B, S, D) + attention output (B, S, D) -> (B, S, D)`

#### Layer normalization

##### What is layer normalization?

(TODO: put)

##### Why do we do layer normalization

(TODO: put)

##### Shapes before/after layer normalization

Input: (B, S, D) -> Output (B, S, D)

### Positionwise feedforward

## Decoder layer

### Decoder masked self-attention

(what is it? Why do we do it? Any analogies?)

### Decoder cross-attention

(what is it? Why do we do it? Any analogies?)
