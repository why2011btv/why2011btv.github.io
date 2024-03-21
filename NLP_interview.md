# Flash Attention
  - [Tutorial](https://youtu.be/FThvfkXWqtE?t=793)
  - [Paper](https://arxiv.org/pdf/2205.14135.pdf)

# Sparse Upcycling: MoE
![image](https://github.com/why2011btv/why2011btv.github.io/assets/32129905/2823bb08-8ec1-4b83-9f50-0ab0f83c662b)

# Inverse Toxicity Filters
  - [Guide](https://arxiv.org/pdf/2305.13169.pdf#:~:text=Inverse%20toxicity%20filters%2C%20which%20remove,toxic%20content%2C%20demonstrate%20targeted%20benefits.)
  - Quality and Toxicity Filters (Section 5). Filtering for document quality and toxicity have significant but
opposite effects on model behaviour. Quality filtering, removing low-quality text, substantially increases both
toxic generation and downstream performance across tasks we tested, despite reducing the amount of training
data. On the other hand, removing toxic data trades-off fewer toxic generations for reduced generalization
performance. Inverse toxicity filters, which remove the least toxic content, demonstrate targeted benefits.
Lastly, evaluation on datasets with high quality text aren’t necessarily improved by removing low-quality
text from the dataset

# [On Determinism](https://community.openai.com/t/a-question-on-determinism/8185)
  - temperature
  - There’s inherent non determinism in GPU calculations around floating point operations - the differences in log probabilities are tiny, but when there’s a small difference between the top two likely tokens, then a different token might be chosen every now and then leading to different results
  - There are speed tradeoffs, and in order to make the endpoints fast GPUs are used, which do parallel (non deterministic) calculations. Any modern gpu neural net calculations will be subject to these.
  - Very simplified example to illustrate the point: a * b * c can be calculated either as (ab) c, or a(bc), but tiny differences can occur when performing floating point operations with the last few significant digits,leading to a very slightly different result. Sometimes these tiny differences can compound and be amplified within a network with argmax on the next token, if the logprobs are very close.

# Llama 2
  - Tokenization: it employs a **bytepair encoding (BPE)** algorithm (Sennrich et al., 2016) using the implementation from **SentencePiece** (Kudo and Richardson, 2018). As with Llama 1, we split all numbers into individual digits and use bytes to decompose unknown UTF-8 characters. The total vocabulary size is **32k tokens**.
  - Context length: 4096 (doubled the context length of the model)
  - grouped-query attention (GQA): Bigger models — 34B and 70B — use Grouped-Query Attention (GQA) for improved inference scalability.
  - SFT 
    - We found that SFT annotations in the order of **tens of thousands** was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations. 
    - For supervised fine-tuning, we use a cosine learning rate schedule with an initial learning rate of 2 × 10−5, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens.
    - A special token is utilized to separate the prompt and answer segments.
    - We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens.
    - Finally, we fine-tune the model for 2 epochs.
  - Reward Modeling 
    - the two responses to a given prompt are sampled from two different model variants, and varying the temperature hyper-parameter
    - We collected a large dataset of over 1 million binary comparisons based on humans applying our specified guidelines, which we refer to as Meta reward modeling data.
    - To address this, we train two separate reward models, one optimized for helpfulness and another for safety
  - RLHF (PPO + Rejection Sampling fine-tuning)
    - We therefore trained successive versions for RLHF models, referred to
here as RLHF-V1, . . . , RLHF-V5.
# LLaMA
  - Context Length: 2048
  - Architecture (main difference with the original transformer architecture (Vaswani et al., 2017), and where we were found the inspiration for this change (in bracket))
    - Pre-normalization [GPT3]; benefit: improve training stability; RMSNorm
    - SwiGLU activation function [PaLM]
    - Rotary Embeddings [GPTNeo]
  - Optimizer
    - AdamW
    - cosine learning rate schedule (the final learning rate is equal to 10% of the maximal learning rate)
    - weight decay (prevent the weights from becoming too large and potentially overfitting the model; works by adding a penalty to the loss function; Specifically with L2 regularization, the penalty is proportional to the square of the magnitude of the weights)
    - gradient clipping (Gradient clipping manages the exploding gradient problem by putting a maximum limit or threshold on the value of the gradient. If a gradient exceeds this threshold, it is set to the threshold. This limits the maximum size of the weight updates (i.e., the steps taken in the optimization algorithm such as stochastic gradient descent), and can lead to more stable and successful learning. It is commonly used when training recurrent neural networks (RNNs), where the exploding gradient problem can be particularly prevalent)
    - 2,000 warm-up steps


# 手搓beam search
The goal of beam search is to improve the quality of the predictions by keeping the most promising candidates at each step. So, instead of keeping the most probable prediction at each step (like in greedy decoding), we keep a fixed number of `beam_size` most probable sequences at each step.

The function `beam_search` implements this process with a language model (`LM`). The input to this function is:

- `LM`: a language model (a function) which takes as input a sequence of token ids (as torch tensor) and outputs a probability for each possible next token in the vocabulary.
- `start_token_id`: the id of the start token in the vocabulary, which is used to start all sequences.
- `end_token_id`: the id of the end token in the vocabulary, which is used to indicate the end of a sequence.
- `max_length`: the maximum length of the sequences. If a sequence reaches this length, it's not extended further.
- `beam_size`: the number of most probable sequences to keep at each step.

Yes, the prompt should be a sequence of tokens. If the prompt is "Once upon a time", you'd have to convert it to token IDs as per your language model's vocabulary, append the start token at the beginning, and then pass it to this function.

Here's an updated version of the function:

```python
def beam_search(LM, prompt_token_ids, end_token_id, max_length, beam_size):
    initial_seq = torch.tensor(prompt_token_ids)
    beams = [(initial_seq, 0)]

    while True:
        new_beams = []
        for (seq, log_prob) in beams:
            if seq[-1] == end_token_id or len(seq) == max_length:  # Don't extend this sequence further
                new_beams.append((seq, log_prob))
            else:
                seq = seq.unsqueeze(0)  # Add batch dimension
                probs = torch.log_softmax(LM(seq), dim=-1)  # Run through LM
                top_probs, top_ids = probs[0, -1].topk(beam_size)  # Get top k probs & ids
                for i in range(beam_size):
                    next_seq = torch.cat((seq, top_ids[i:i+1].unsqueeze(0)), dim=-1)
                    next_prob = log_prob + top_probs[i].item()
                    new_beams.append((next_seq, next_prob))

        # Sort all available beams by score and keep the best `beam_size` ones
        new_beams.sort(key=lambda tup: -tup[1], reverse=True)
        beams = new_beams[:beam_size]
        if all(seq[-1] == end_token_id for seq, _ in beams):  # All sequences have ended
            return beams
```
# 手搓transformer
Let's assume the following scenario for our multi-head attention implementation:

- `d_model` (Input Dimension): 512
- Number of heads `h`: 8
- `d_k` (`d_model` / `h`): 64
- Batch size: 64
- Sequence Length: 200

So, we have:

- Input vectors (Query, Key, Value): [64 (Batch Size), 200 (Seq Length), 512 (`d_model`)]
- After passing through the Linear Layers (`self.w_q`, `self.w_k`, `self.w_v`), the dimensions remain the same: [64, 200, 512]
- These are then reshaped and transposed for multi-head attention, leading to: [64 (Batch Size), 8 (Number of Heads), 200 (Seq Length), 64 (`d_k`)]
- The attention scores (the result of Q and K dot product) will be: [64, 8, 200, 200]
- The output after the attention score applied to V is: [64, 8, 200, 64]
- Finally, after combining the heads and passing through the output linear layer (`self.fc_out`), we get: [64, 200, 512]

If a mask is applied, it should be of shape [64, 1, 1, 200] to correspond to the attention scores' shape. It is applied broadcasted along the third dimension when calculating the attention scores.

This flow lets each head learn different types of attention (e.g., one head might pay attention to the previous word, another to the subsequent word), and allows for more complex interactions between words.
```python
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        N = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_model**0.5
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(self.dropout(attention), V)
        
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.d_model)

        out = self.fc_out(out)

        return out
```
The image you've shared shows two Transformer architecture variants: (a) Post-Layer Normalization (Post-LN) and (b) Pre-Layer Normalization (Pre-LN). The architectures are used for building deep learning models, especially for tasks like language understanding and translation. 

Writing complete Python code for these architectures from scratch can be quite involved, but I can give you a high-level example using PyTorch, a popular deep learning framework.

```python
import torch
import torch.nn as nn

# Define the Multi-Head Attention block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, query, key, value, mask):
        # Forward pass of multi-head attention
        attn_output, _ = self.attention(query, key, value, mask)
        return attn_output

# Define the Feedforward block
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Define the Transformer Block for Post-Layer Normalization (Post-LN)
class TransformerBlockPostLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output = self.attention(x, x, x, mask)
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

# Define the Transformer Block for Pre-Layer Normalization (Pre-LN)
class TransformerBlockPreLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, mask):
        x = self.norm1(x)
        attn_output = self.attention(x, x, x, mask)
        x = x + attn_output
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x
```

In the code above:
- `d_model` is the dimensionality of the model (typically 512 or 768 in BERT).
- `num_heads` is the number of attention heads (e.g., 8 or 12).
- `d_ff` is the dimensionality of the feed-forward network's inner layer (typically 2048).
- `mask` is used to ignore padding tokens in the input during the attention operation.

You can instantiate a transformer block using:

```python
post_ln_transformer_block = TransformerBlockPostLN(d_model=512, num_heads=8, d_ff=2048)
pre_ln_transformer_block = TransformerBlockPreLN(d_model=512, num_heads=8, d_ff=2048)
```

Remember, this is a very high-level example and lacks many details such as proper mask handling, dropout, and other nuances. For a full implementation, you would typically use an existing library like `transformers` from Hugging Face.



# complexity of transformer

# long-context how?

# Tokenizer: word-level, character-level, subword-level

## WordPiece: BERT, DistilBERT<img width="1662" alt="Screenshot 2024-01-13 at 5 24 35 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/aa48a4cf-67b0-428c-af4d-5edca60a44d7">
## Unigram / SentencePiece: XLNet, ALBERT, T5, mBART
  - 删除token 使得unigram loss增加得最少<img width="1509" alt="Screenshot 2024-01-13 at 5 49 59 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/abc0b5bc-ace0-4776-82d0-cdd8ce2eaeca">
## Byte-Pair Encoding / BPE / BBPE: GPT-2, RoBERTa, GPT-J, LLaMA
  - 词频统计 + 词表合并<img width="1638" alt="Screenshot 2024-01-13 at 5 20 14 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/fda20ba0-e31c-41b6-9251-bf37ced6a681">
  - <img width="1618" alt="Screenshot 2024-01-13 at 5 22 40 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/3108351e-5313-4e8f-b879-e671c7b214fc">



# Positional Embedding
## sinusoidal embeddings
## learned embeddings

# Attention mechanism
## <img width="360" alt="Screenshot 2024-01-13 at 3 36 27 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/446b3e8c-9e03-48b6-bac0-b77d0cf7da76">
## <img width="510" alt="Screenshot 2024-01-13 at 3 37 34 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/8470d694-38fa-451f-8fe5-9ec1e9ff3028">
## why projection for Q, K, V? (size_per_head = H, [B, L, D] -> [BxL, NxH])
  - For specialization. Different projections allow the model to learn different aspects of the information present in the input.
  - Q, K, and V perform different roles within the attention mechanism — queries are used to score against keys, keys define the addressable contents for attention, and values contain the actual content that will be focused on after attention scores are computed.
  - Having distinct representations for each allows the model to optimize for these different roles.
  - **Without projection, when calculating attention, we will use the dot product of Q and K (which are the same), and then get very high scores on the diagonal elements in the attention matrix, thus not combining semantic / syntactic info from context**.
## why multiple heads? (num_attention_heads = N)
  - We hope that different attention head can learn different representation
## shape of attention_scores: [B, N, L, H] x [B, N, H, L] -> [B, N, L, L]
## Why do we use dot-product (multiplicative) attention instead of additive attention?
  -  While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
## why do we scale it with 1/sqrt(size_per_head)?
  - <img width="1174" alt="Screenshot 2024-01-13 at 4 11 00 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/20bdfebe-72a5-489c-b8b0-6dacbbffebd7">
## attention_mask: [B, L, L]; usage: training (1, 1, ..., 1, 0, 0) and inference (upper right corner are all 0, since we cannot see the future)
  - adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0
## Add & Norm![Screenshot 2024-03-20 at 12 15 20](https://github.com/why2011btv/why2011btv.github.io/assets/32129905/8489c466-708d-4591-a319-a97b4fd97c78)

  - <img width="1162" alt="Screenshot 2024-01-13 at 4 27 14 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/ccab4f67-b074-466c-8133-926aa45d93c6">
  - Add (residual connection): 相当于在求导时加了一个恒等项，去减少梯度消失的问题
  - Batch-norm vs Layer-norm<img width="1103" alt="Screenshot 2024-01-13 at 4 32 57 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/c908cc29-0793-4fcc-8881-b6de25d08a7d">
  - 序列任务中更常用Layer-norm：因为序列数据的长度不一样，batch-norm在针对不同样本的同一位置做归一化时无法得到真实分布的统计值；而layer-norm会对同一个样本的每一个位置的不同特征都做归一化
  - Order of Add & Norm: 保持主干网络的方差稳定，使模型泛化能力更强，但不容易收敛；如果先Norm后Residual（pre-normalization）：只是增加了网络宽度，深度没有太大增加，效果不如post-normalization好
  - [Post-LN vs Pre-LN](https://arxiv.org/pdf/2002.04745.pdf)
    -  the scale of the expected gradients grows along with the layer index for the Post-LN Transformer. On the contrary, the scale almost keeps the same for different layers in the Pre-LN Transformer
    -  Pre-LN: the learning rate warm-up stage can be safely removed, and thus, the number of hyper-parameter is reduced.
    -  Furthermore, we observe that the loss decays faster for the Pre-LN Transformer model. It can achieve comparable final performances but use much less training time  
## FFN Activation function:
  - ReLU (Attention is all you need)
  - GeLU (BERT): Introduce regularization; 越小的值越容易被丢弃；相当于ReLU和dropout的综合
  - why not tanh or sigmoid: 这两个函数的双边区域会饱和，导致导数趋于0，有梯度消失的问题，不利于深度网络的训练

# Training objectives
## MLM + NSP
## Language Model

# Architecture
## Encoder: BERT
## Decoder: GPT 
## Encoder-Decoder: T5 (trained on C4)

# Decoding
## Temperature
温度参数会对softmax函数的输出产生如下影响：在应用softmax函数时，通常针对每个可能的下一个词计算一个分数，这些分数被用来获得一个概率分布，根据这个分布选取下一个词。softmax函数的公式是:
$$ \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$
其中，\(x_i\) 表示模型输出的原始分数（logits）中的第i个分数。
Temperature \(T\) 被引入到softmax函数中以调整分数：
$$ \text{softmax}_T(x_i) = \frac{\exp(x_i/T)}{\sum_j \exp(x_j/T)} $$
  - 当 temperature \(T=1\) 时，这是标准的softmax函数。
  - 当 temperature \(T>1\) 时，softmax输出的概率分布会变得更加平滑或"软"，增加了生成过程中的随机性，也就是模型在选择下一个词时更加不确定，可能生成更多样化的文本。
  - 当 temperature \(T<1\) 时，概率分布变得更加"尖锐"，即减少随机性，模型更可能选择最高概率（即最可能的）词汇，可能生成更确定性、可预测性强的文本。
  - 当 temperature 接近 0 时，这个过程变成近似贪婪搜索，总是选择概率最高的词汇。
## **Top-K Sampling**: 这种方法在每一步将选择的可能性限定在最高K个概率的词中。通过只考虑概率最高的K个词，模型避免生成低概率的词汇，从而可以提升生成结果的质量。

## **Top-p（Nucleus）Sampling**: 与 Top-K 不同，Top-p Sampling 只选择累积概率达到预设阈值p（例如0.9）的词的集合来抽取下一个词。这允许动态选择词汇范围大小，并可以生成更加丰富和不可预测的文本。

## **Beam Search**: Beam Search 是一种启发式搜索算法，它在解码过程中跟踪多个可能的候选序列（称为"beams"）。参数“beam width”（也称为beam size）决定了同时探索的序列数。此算法最终选择具有最高整体概率的序列作为输出。

## **Length Penalty**: 在Beam Search 中，长度惩罚参数通过减弱或加强较长序列的概率分数，帮助平衡解码过程中生成结果的长度。这有助于控制生成的文本片段的长度。

## **Repetition Penalty**: 通过惩罚重复词汇的分数，降低重复词汇出现的概率，帮助生成更多样化的文本。

## **Minimum/Maximum Length**: 设置生成文本的最小和最大长度限制，可以控制生成文本的长度范围。

7. **Early Stopping**: 设置是否在满足某个条件时提前停止生成过程，如遇到特定的终止符。

8. **Presence Penalty / Frequency Penalty**: 这些惩罚项用于降低或提高预先指定的词语出现在生成文本中的频率，进一步控制文本输出的多样性。
