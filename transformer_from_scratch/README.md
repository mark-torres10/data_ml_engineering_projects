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

We then add positional encodings. This lets the model know where each token occurs in the sequence

(explain in more detail what problem positional encodings solve)

(dummy code for what an example positional encoding looks like)

```python

```

```python
x = ""
print(x.shape) # (1, 8, 512)
```

...

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
