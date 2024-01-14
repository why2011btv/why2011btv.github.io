# Tokenizer
## WordPiece: BERT, DistilBERT<img width="1662" alt="Screenshot 2024-01-13 at 5 24 35 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/aa48a4cf-67b0-428c-af4d-5edca60a44d7">
## Unigram / SentencePiece: XLNet, ALBERT, T5, mBART
  - 删除token 使得unigram loss增加得最少<img width="1509" alt="Screenshot 2024-01-13 at 5 49 59 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/abc0b5bc-ace0-4776-82d0-cdd8ce2eaeca">

## Byte-Pair Encoding / BPE / BBPE: GPT-2, RoBERTa, GPT-J, LLaMA
  - 词频统计 + 词表合并<img width="1638" alt="Screenshot 2024-01-13 at 5 20 14 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/fda20ba0-e31c-41b6-9251-bf37ced6a681">
  - <img width="1618" alt="Screenshot 2024-01-13 at 5 22 40 PM" src="https://github.com/why2011btv/why2011btv.github.io/assets/32129905/3108351e-5313-4e8f-b879-e671c7b214fc">


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
