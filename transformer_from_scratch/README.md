# Implementing a basic version of the original encoder/decoder transformer from scratch

Our goal is to implement the following architecture from the original transformer:

![Original Transformer Architecture](static/the-annotated-transformer.png)

This folder contains a workthrough of Karpathy's [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY). The associated GitHub repo for the original implementation is [in this link](https://github.com/karpathy/nanoGPT)

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

### The original context for the encoder + decoder system

The original transformer model was developed at Google specifically for the task of **machine translation**. In this problem, the model receives a complete source sentence and it must translate it to the next sentence.

To translate a text requires doing the following:

1. Understanding the input sentence.
2. Translating the input sentence.

You can't literally just translate word-for-word. Imagine, for example, the following two sentences:

- "He went to the river bank to catch fish"
- "He went to the bank to withdraw money"

Here, you can't do a literal translation of the word "bank" independent of the other words in the sentence. You must understand the word "bank" in relation to the other words.

We need to first understand the sentence and then make the translation based on the understanding. We need to understand what each word means *in context* to the other words in the sentence.

This perfectly maps to the encoder/decoder setup:

1. Our encoder understands the input sentence. It then passes its understanding to the decoder.
2. Based on the encoder's understanding of the sentence, the decoder produces the translation.

### Explaining the encoder + decoder architecture through analogy

The encoder + decoder architecture splits this task into two parts, understanding and generation. The encoder has to analyze the entire sentence all at once and generate a "this is what the entire sentence means" representation of the sentence. Specifically, each token in the sentence is reweighted or transformed based on its meaning in the context of the whole sentence. This is called **self-attention**. In contrast, the decoder combines two forms of attention: **casual self-attention**, where the decoder looks at and understands what it's generated so far, and **cross-attention**, where the decoder then looks at the encoder's output and pulls what info it needs for additional context.

To understand the different forms of attention and what the encoder and decoder do, let's use an overly simplified analogy:

Let's say that we're working at a media tech startup as software engineers, and our customers want to create a "summarize the headlines" feature on their website, so that customers can always get an updated report of what's going on in the world that's relevant to them.

- We have a product team that talks to the customer. They get the initial specs, feedback, requirements, and budget from the customers. They translate this into a synthesized understanding, based on their knowledge of the clients (e.g., "this requirement is actually not as important as this other requirement, or this other requirement wasn't emphasized enough but I know they'll actually care about it"). They may do this for a few rounds back-and-forth with the customer. They then hand off a structured set of notes, where each requirement has been rewritten and annotated in the context of all the others. For example, instead of saying "use the most powerful AI model" (which was an initial requirement), the version handed off instead says "use the most powerful AI model... which fits XYZ budget" (combining two separate requirements, "use the most powerful AI model", and "here's our budget"). Importantly, every requirement is rewritten in light of the other requirements, to give a series of interdependent, context-aware requirements.
- The engineering team then takes this requirement and starts building, one day at a time. Let's say it's just one software engineer building features one day at a time. They base what they build by looking first at "what did I build the previous days?", get an understanding of what they think they need to work on today, and then peek at the product team's notes and specs are, and then refine their understanding. They might do this for a few rounds (look at their previous code, then look at the product specs, then look at the existing code again, a few times). Finally, they figure out "this is what I should do next", and build the next part of the code.

In this analogy, the product team is like the encoder of the original transformer model. They get an "understanding" of what the original input was (here, the input being the client specs and requirements) and then give that translated understanding to the decoder.

- They may take the client specs and update them by incorporating parts of the other inputs as context (e.g. the client might say "I want to use the most powerful AI model", but they also said "we have a budget of XYZ", so you reinterpret the initial ask as "I want to use the most powerful AI model... that is within our budget"), This is like the **self-attention** mechanism of the encoder block. We transform the meaning of a single token (here, "I want to use the most powerful AI model") in the context of the entire sequence (e.g,. "we have a budget of XYZ") to create a transformed representation of the single token in context (e.g., "I want to use the most powerful AI model... that is within our budget"). Importantly, we want to keep all the requirements that the client gave (we can't just ignore a request that the client explicitly gave) but we reweigh them in light of other requirements. For example, the client said they both want the most powerful AI *and* to stick within a certain budget, but it's probably more important we stick to the budget, so we give that requirement more weight, but we don't ignore the "most powerful AI" requirement.
- Then based on this reweighting of the specs, the product team can reword certain requirements (e.g., if a bunch of customers pointed out that they were concerned about privacy, we can turn that initial requirement into a new requirement, like "make sure that we have multitenancy and encryption available"), reinterpret them, add certain heuristics (e.g., "whenever the client says that they want it in 3 weeks, we know that we can realistically do it in 6 weeks, so let's update the requirement to be delivered in 6 weeks instead"), and rewritten to a more precise and structured format (e.g., the client said "we want the most powerful AI", let's rewrite that to say "let's use XYZ specific model version"). This is like the feedforward neural network portion of the encoder. It is in the FFNN that the actual "thinking" happens (most parameters in a transformer are in the FFNN).
- Importantly, the FFNN layer is position-wise, so the tokens don't interact with each other. For our analogy, imagine that each requirement in the specs gets sent to a specialist (e.g., an accountant, a security analyst, etc.). Each specialist looks at the requirement they've been given (e.g., "we need GDPR compliance"), expands on it ("what does it take to do GDPR compliance"), applies their own reasoning (e.g. "how long would it take to actually have GDPR compliance, and can we do it in a V1?"), and compresses it into a refined requirement (e.g., client says "we need GDPR compliance" and "we want the result in 3 weeks", and the requirement is rewritten as "we can have XYZ portions of GDPR compliance by ABC weeks if we do DEF requirements").
- They may have to do this for a few rounds of iteration (the product team has the v1 draft of their understanding of the client specs, then they might revise it a few times), representing the `N` blocks that the encoder block may have. Once they're satisfied with this understanding, they pass this off to the engineering team.

In this analogy, the engineering team is like the decoder of the original transformer model. They take (1) their existing codebase and (2) what specs they got from the product team and then make the feature come to life. Let's imagine what one engineer does on a given day.

- First, the engineer looks at the code already written for the project. This represents the **causal self-attention** mechanism. The engineer looks at the already-generated code to understand where they are right now. In a decoder, a causal mask prevents the decoder from "looking ahead" into the future; in this analogy, an engineer cannot "look ahead" to code that doesn't literally exist yet.
- Then, the engineer looks at the specs from the product team and refines their understanding of what they need to do today. This is **cross-attention**; the decoder queries the encoder's keys/values to get the additional context they need to do the generation task. In the same way, the engineer asks targeted questions ("queries") to the product specs to get more information based on what they're trying to build (e.g., the engineer may want to know "which libraries are we allowed to use?" and then check for this information in the product spec). The engineer, like the product team, can't just ignore any of the client requirements, so they must take it all into consideration (e.g., "blending"). But they can assign weight on which requirements are relevant for *today's* work. For example, the client may have said something about "having the most powerful AI", but if the engineer is just doing frontend today, then the "having the most powerful AI" isn't really relevant to today's code (but importantly, they still read it and keep it in the back of their mind). They mentally highlight which parts of the requirements are *most relevant* for today's work, while keeping everything else in the back of their mind.
- The engineer then takes what it learned from looking at what's been generated and what the client specs are and then synthesizes that information to create their own understanding of what it needs to do. This is like the feedforward layer of the decoder block, where the decoder takes the understanding from the causal self-attention and the cross-attention and builds a new representation of the context. Here, this is done position-wise again, meaning, e.g., the engineer can reinterpret the requirements of each function and unit of work independently of the others. For example, the engineer considers what they need to build for the auth flow separately from the toggle button, but they're all part of today's unit of work.
- At each thinking pass, the engineer (1) considers the past work (self-attention), (2) considers specs (cross-attention) and (3) updates their internal understanding and plan. They can do this `N` times, representing the `N` passes through the decoder block. They do this before they write any code.
- Finally, based on multiple rounds of understanding what they're supposed to do, the engineer can generate the code. This represents the actual linear and softmax layers of the transformer.
- The next day, the engineer can log on, and take the exact same product specs (the same output from the decoder), plus what they produced today (from the decoder + linear + softmax) and then continue with their "autoregressive code generation" task.

This analogy also highlights one key difference between the encoder and decoder steps. The product team (encoder) processes all requirements at once, while the engineer (decoder) works step-by-step, building one piece at a time. In a similar way, the encoder is fully parallel while the decoder is autoregressive.

### Encoder-only architectures

### Decoder-only architectures

### The shift to mixture-of-experts (MoE)

### The KV cache

### Speculative decoding

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
