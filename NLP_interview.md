# LLaMA
  - Architecture (main difference with the original transformer architecture (Vaswani et al., 2017), and where we were found the inspiration for this change (in bracket))
    - Pre-normalization [GPT3]; benefit: improve training stability; RMSNorm
    - SwiGLU activation function [PaLM]
    - Rotary Embeddings [GPTNeo]
  - Optimizer
    - AdamW
    - cosine learning rate schedule (the final learning rate is equal to 10% of the maximal learning rate)
    - weight decay (prevent the weights from becoming too large and potentially overfitting the model; works by adding a penalty to the loss function; Specifically with L2 regularization, the penalty is proportional to the square of the magnitude of the weights)
    - gradient clipping (Gradient clipping manages the exploding gradient problem by putting a maximum limit or threshold on the value of the gradient. If a gradient exceeds this threshold, it is set to the threshold. This limits the maximum size of the weight updates (i.e., the steps taken in the optimization algorithm such as stochastic gradient descent), and can lead to more stable and successful learning. It is commonly used when training recurrent neural networks (RNNs), where the exploding gradient problem can be particularly prevalent)


# 手搓beam search

# 手搓transformer

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
## Add & Norm
  - <img width="1162" alt="Screenshot 2024-01-13 at 4 27 14 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/ccab4f67-b074-466c-8133-926aa45d93c6">
  - Add (residual connection): 相当于在求导时加了一个恒等项，去减少梯度消失的问题
  - Batch-norm vs Layer-norm<img width="1103" alt="Screenshot 2024-01-13 at 4 32 57 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/c908cc29-0793-4fcc-8881-b6de25d08a7d">
  - 序列任务中更常用Layer-norm：因为序列数据的长度不一样，batch-norm在针对不同样本的同一位置做归一化时无法得到真实分布的统计值；而layer-norm会对同一个样本的每一个位置的不同特征都做归一化
  - Order of Add & Norm: 保持主干网络的方差稳定，使模型泛化能力更强，但不容易收敛；如果先Norm后Residual（pre-normalization）：只是增加了网络宽度，深度没有太大增加，效果不如post-normalization好
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
