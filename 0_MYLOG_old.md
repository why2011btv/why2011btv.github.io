## 01/09/2024
- Web Agent Benchmarks
  - AssistantBench [2407.15711](https://arxiv.org/pdf/2407.15711)
    - realistic and time-consuming tasks on the web, e.g., monitoring real-estate markets or locating relevant nearby businesses
  - Androidworld: A dynamic benchmarking environment for autonomous agents [2405.14573](https://arxiv.org/pdf/2405.14573)
  - OSWorld [2404.07972](https://arxiv.org/pdf/2404.07972)<img width="1085" alt="Screenshot 2025-01-09 at 12 06 58 PM" src="https://github.com/user-attachments/assets/789e26b2-8d5f-4a2c-94bc-ea626ae7bd0f" />
  - VisualWebBench: Web Page Understanding and Grounding [2404.05955](https://arxiv.org/pdf/2404.05955)
    - seven tasks, and comprises 1.5K human-curated instances from 139 real websites, cover- ing 87 sub-domains
  - Tur[k]ingBench: A Challenge Benchmark for Web Agents [2403.11905](https://arxiv.org/pdf/2403.11905)
    - 158 web-grounded tasks; Amazon Mechanical Turk Tasks<img width="1314" alt="Screenshot 2025-01-09 at 11 35 03 AM" src="https://github.com/user-attachments/assets/c59035e7-9b6d-486b-a367-943cd34d40dc" />
  - WorkArena a benchmark developed on the widely-used ServiceNow platform [2403.07718](https://arxiv.org/pdf/2403.07718)
  - WebLINX: Real-World Website Navigation with Multi-Turn Dialogue [2402.05930](https://arxiv.org/pdf/2402.05930)
    - given the initial user instruction, an agent must complete a real-world task inside a web browser while communicating with the user via multi-turn dialogue<img width="1446" alt="Screenshot 2025-01-09 at 11 34 20 AM" src="https://github.com/user-attachments/assets/805927c1-7e26-4bb3-bb6c-17d7201cc06c" />
  - WebVoyager [2401.13919](https://arxiv.org/pdf/2401.13919)
    - 300 information-retrieval tasks, from 15 real-world consumer websites (e.g., Amazon, Coursera, Booking)
  - AgentBench [2308.03688](https://arxiv.org/pdf/2308.03688)
    - poor long-term reasoning, decision-making, and instruction following abilities; Training on code and high quality multi-turn alignment data could improve agent performance<img width="587" alt="Screenshot 2025-01-09 at 12 14 49 PM" src="https://github.com/user-attachments/assets/6d6d9cef-76c7-4b25-b7b2-eb96179bd924" />


## 01/03/2024
- Online vs Offline RL
  - [Offline RL](https://youtu.be/tW-BNW1ApN8?si=hnGTMmYYlYd6kyhz)
    - At training time, RL is conventionally viewed as an active and online process where an agent interacts with the world, collects some experience, uses that experience to modify its behavior which we call policy, and then collects some more experience and this is done many, many times. "Learning through trial and error"
    - The assumptions of the offline RL process do not require you to do this more than once (collect a dataset using any policy or mixture of policies)
    - Then step two is to run an offline RL algorithm on this dataset and intuitively what this algorithm will do is it will squeeze the best behavior it can out of that data. So you can think of it as the best policy we can get based on what the data tells us.
    - The end goal in offline RL is to get a better policy than the one that collected your data.
    - Fundamental problem: counterfactual queries
      - the policy is trying to figure out what would happen if it were to do something other than what was done in the data
      - fundamentally if you want to improve the behavior in the data you have to answer these counterfactual queries
      - <img width="1704" alt="Screenshot 2025-01-03 at 4 25 35 PM" src="https://github.com/user-attachments/assets/5104e660-fef7-44f3-ad03-6aa278f69dfd" />


## 01/02/2024
- Deepseek V3
  - load balancing
  - multi-token prediction
  - fp8 mixed precision training
  - cross-node MoE training
  - multi-head latent attention
  - <img width="871" alt="Screenshot 2025-01-02 at 5 08 16 PM" src="https://github.com/user-attachments/assets/0afaeb2e-a1e6-4401-8b55-2d039747ca8c" />
  - RMSNorm vs LayerNorm: 省略了均值计算，减少了计算开销，适合用于加速模型训练
  - <img width="859" alt="Screenshot 2025-01-02 at 5 03 57 PM" src="https://github.com/user-attachments/assets/626e93ea-059f-49b2-af24-bd659d4d41bb" />

- Attention is all you need 
  - [Visualizing transformers and attention](https://youtu.be/KJtZARuO3JY?si=mPS02oDnMM__JanF)
  - "The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key"
  - Q:
    - "creature": Any adjectives in front of me?
  - K:
    - "fluffy": I am!
    - "blue": I am!
  - softmax(Q K^T / sqrt(d_k)): **attention_scores** (a compatibility function of the query with the corresponding key)
  - softmax(Q K^T / sqrt(d_k)) V: a weighted sum of the values
  - <img width="1031" alt="Screenshot 2025-01-02 at 4 25 37 PM" src="https://github.com/user-attachments/assets/e5660e98-c358-42f0-92ae-03bb038479ce" />
  - <img width="1020" alt="Screenshot 2025-01-02 at 4 23 35 PM" src="https://github.com/user-attachments/assets/589c8bbd-48a2-40e7-a134-c6d265a0b8ab" />
  - <img width="892" alt="Screenshot 2025-01-02 at 4 57 10 PM" src="https://github.com/user-attachments/assets/c19f811b-2af6-4576-87b6-9d36e7550aed" />



## 12/30/2024
- [Large Concept Model](https://scontent-mia3-2.xx.fbcdn.net/v/t39.2365-6/470149925_936340665123313_5359535905316748287_n.pdf?_nc_cat=103&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=eVE4IWzjoAkQ7kNvgHXBt-7&_nc_zt=14&_nc_ht=scontent-mia3-2.xx&_nc_gid=AQn_G_S3xH05xScryOReG6K&oh=00_AYBS2dRAJAGYmtEwGyAyr4zJ0B1QFk0yGhxWnnspHCB8iQ&oe=6778C492)
- [ClariQ Dataset - EMNLP'21](https://aclanthology.org/2021.emnlp-main.367.pdf)
- [Agent-CQ](https://arxiv.org/pdf/2410.19692)
- [Travel Planning Agent - Meta, Nov. 2024](https://arxiv.org/pdf/2411.13904)
- [Travel Planning Mixed Integer Linear Programming Solver - Meta, Oct. 2024](https://arxiv.org/pdf/2410.16456)
- [Coordinate Descent for LASSO](https://youtu.be/afMcBZgpauM?si=FMoELQjfNc6amuLN)
- [Planning in NL for Code Gen - Scale AI, Sep. 2024](https://arxiv.org/pdf/2409.03733)
- 
- 

## 06/26/2022
- [Latex Symbols](https://www.caam.rice.edu/~heinken/latex/symbols.pdf)

## 02/17/2022
- Tense detection
  - https://tense-sense-identifier.herokuapp.com
    - curl --request POST --data '{"data": "She would say the soldiers were hit by a truck."}' -H "Content-type: application/json" https://tense-sense-identifier.herokuapp.com/home
  - https://u-aizu.ac.jp/~jblake/course_tense/tense_unit_07.html
## 12/14/2021
- <img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}">
- Now we have tools for estimating uncertainty in neural networks. How can we apply to unanswerable questions?
  - In image classification, there are fixed number of labels, e.g., cat, dog, human, car, and so forth. Out-of-distribution testing examples can be unseen categories in the training data, or foggy / blurry pictures. 
  - In extractive Question Answering, enumerating over all possible spans of the context passage is computationally costly. Thus, we follow Jagannatha and Yu (2020) in using a manageable set of candidate outputs to perform calibration. We finally keep the top K spans as candidates I(X) and use all candidates to calculate the normalized probability which provides some idea of the confidence of answer <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}"> with respect to the candidate list.
  - Calibration: Fixed set of answers in image classification VS Changeable answers in QA
    - We approximate this probability by bucketing predictions into M disjoint equally-sized interval bins based on confidence.
    - It doesn't matter whether the outputs are fixed categories or changeable answers (spans from different input context). As long as the model outputs probabilities for different candidates, ECE can be calculated.
    - For example, there are 3 cases for prediction, the probability predicted for the correct answer is 0.3, 0.6, 0.9; whereas we split into 2 buckets, 0-0.5 and 0.5-1. 
  


## 12/13/2021
- Predictive Uncertainty Estimation via Prior Networks
  - Model Uncertainty: given training data D, the probability of finding the perfect model \theta for the task
    - Cause: how well the model is matched to the training data
  - Distributional Uncertainty: given an example (test) input x* and the perfect model $\theta$, the probability of getting the perfect categorical distribution $\mu$ over class labels
    - Cause: difference between training and test data
  - Data Uncertainty: given the categorical distribution, the probability of getting the right prediction. 
    - Cause: class overlap, label noise
  - <img width="1001" alt="Screen Shot 2021-12-13 at 11 53 48 PM" src="https://user-images.githubusercontent.com/32129905/145935431-43cc7f59-b0af-450c-bb98-52efe42f04b2.png">

## 11/08/2021
- Google mail smtp 
  - [less secure apps](https://www.google.com/settings/security/lesssecureapps)

## 11/03/2021
- Controlled Generation
  - [A Hybrid Model for Globally Coherent Story Generation](https://aclanthology.org/W19-3404.pdf)
  - [Constrained Labeled Data Generation for Low-Resource Named Entity Recognition](https://aclanthology.org/2021.findings-acl.396.pdf)
  - [A General-Purpose Algorithm for Constrained Sequential Inference](https://aclanthology.org/K19-1045.pdf)
  - [Learning to Write with Cooperative Discriminators](https://aclanthology.org/P18-1152.pdf)
  - [Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints](https://arxiv.org/pdf/1809.01215.pdf)
  - [Hafez: an Interactive Poetry Generation System](https://aclanthology.org/P17-4008.pdf)
  - [Controllable Neural Text Generation](https://lilianweng.github.io/lil-log/2021/01/02/controllable-neural-text-generation.html)


- Schema Induction
  - [Machine-Assisted Script Curation](https://aclanthology.org/2021.naacl-demos.2.pdf)
  - [Connecting the Dots: Event Graph Schema Induction with Path Language Modeling](https://aclanthology.org/2020.emnlp-main.50.pdf)
  - [Future is not One-dimensional: Graph Modeling based Complex Event Schema Induction for Event Prediction](https://arxiv.org/pdf/2104.06344.pdf)

- Probing
  - [Probing Natural Language Inference Models through Semantic Fragments](https://arxiv.org/pdf/1909.07521.pdf)
  - [Probing Across Time: What Does RoBERTa Know and When?](https://arxiv.org/pdf/2104.07885.pdf)

- Reasoning
  - [What’s Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering](https://aclanthology.org/D19-1281.pdf)
## 10/30/2021
- Mojave text selection too dark in Preview 
  - [Solution](https://www.techjunkie.com/exclude-app-dark-mode-macos-mojave/)

## 10/19/2021
[.gitignore not ignoring files](https://stackoverflow.com/questions/45400361/why-is-gitignore-not-ignoring-my-files)

## 09/28/2021
- Price for using GPT-J-6B on nlpcloud.io
  - <img width="1792" alt="Screen Shot 2021-09-28 at 3 50 20 PM" src="https://user-images.githubusercontent.com/32129905/135156095-41d273eb-9ae8-4f6b-a202-77ccf520a2f8.png">
- [Variational Bayes](https://blog.evjang.com/2016/08/variational-bayes.html)
- [torch index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html)
- [tf einsum](https://www.tensorflow.org/api_docs/python/tf/einsum)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.html)
- [Wide Narrow Reading of Events](http://cairo.lti.cs.cmu.edu/kbp/2017/event/TAC_KBP_2017_Event_Coreference_and_Sequence_Annotation_Guidelines_v1.1.pdf)
- [Understanding VAE](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [colab TPU GPU](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/GPUvsTPU.ipynb#scrollTo=QNh64VMDz1Ks)
- [haiku](https://github.com/deepmind/dm-haiku)
- [JAX](https://github.com/google/jax)
- [GPT-J-6B](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b)
- [Deep Metric Learning](https://towardsdatascience.com/the-why-and-the-how-of-deep-metric-learning-e70e16e199c0)
- [Label Smoothing](https://arxiv.org/pdf/1906.02629.pdf)
- [Interpretable machine learning](https://christophm.github.io/interpretable-ml-book/taxonomy-of-interpretability-methods.html)
- [CIS 700 Prof. Ungar](https://docs.google.com/document/d/18rCZKLcCp6bssVS2TPd-mHzMxFCGTG5kP3AkUxLk7Yc/edit)
- [What-if I ask you to explain: Explaining the effects of perturbations in procedural text](https://aclanthology.org/2020.findings-emnlp.300.pdf)
- [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://aclanthology.org/P19-1487.pdf)
- [Towards Harnessing Natural Language Generation to Explain Black-box Models](https://aclanthology.org/2020.nl4xai-1.6.pdf)
- [Explaining Simple Natural Language Inference](https://aclanthology.org/W19-4016.pdf)
- [AMR IBM](https://github.com/IBM/transition-amr-parser/blob/master/scripts/README.md#install-details)

## 09/22/2021
- [Understanding VAE](https://arxiv.org/pdf/1907.08956.pdf)
  - x<sub>i</sub>: a data point, e.g., a twitter feed
  - z: latent variable, e.g., the emotion distribution, "happy: 0.3, fear: 0.1, ..."
  - &Theta;: neural network weights (encoder)
  - &phi;: neural network weights (decoder)
  - real posterior (what we want): p(z|x<sub>i</sub>) 
  - approximate of posterior: q<sub>&Theta;</sub>(z|x<sub>i</sub>) 

## 09/21/2021
- [Variational Inference & ELBO](https://blog.evjang.com/2016/08/variational-bayes.html)
  - Given this twitter feed X, is the author depressed (latent variable Z)?
  - The idea behind variational inference is this: let's just perform inference on an easy, parametric distribution Qϕ(Z|X) (like a Gaussian) for which we know how to do posterior inference, but adjust the parameters ϕ so that Qϕ is as close to P as possible.
  - Since KL(Q||P)≥0, logp(x) must be greater than L. Therefore L is a lower bound for logp(x). L is also referred to as evidence lower bound (ELBO)
- [Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
  - So, in order to be able to use the decoder of our autoencoder for generative purpose, we have to be sure that the latent space is regular enough. One possible solution to obtain such regularity is to introduce explicit regularisation during the training process. Thus, as we briefly mentioned in the introduction of this post, a variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.
  - Regularity of latent space:
    -  ![1*83S0T8IEJyudR_I5rI9now@2x](https://user-images.githubusercontent.com/32129905/134374541-b2a996db-f272-410d-a0cb-02072b527dcb.png)



## 09/03/2021
- [BigBird](https://huggingface.co/blog/big-bird)
  - <img width="808" alt="Screen Shot 2021-09-03 at 1 51 47 AM" src="https://user-images.githubusercontent.com/32129905/131957028-d7456023-26e5-4c12-9fcc-89ff7bf4bfd8.png">
  
- raise Exception('My error!')
- [apex](https://github.com/NVIDIA/apex)

## 09/02/2021
- [TACRED Revisited](https://arxiv.org/abs/2004.14855)

## 09/01/2021
- Stats about HiEve & IC
  - max event num in a doc: article-15708.tsvx 102; NYT_ENG_20050312.0073 103
  - max doc_to_TransformerTokenIDs(): 1313; 1533

## 08/16/2021
- Document-level IE
  - [Joint Detection and Coreference Resolution of Entities and Events with Document-level Context Aggregation](https://aclanthology.org/2021.acl-srw.18/)
  - [Document-level Event Extraction via Parallel Prediction Networks](https://aclanthology.org/2021.acl-long.492/)
  - [Document-level Event Extraction via Heterogeneous Graph-based Interaction Model with a Tracker](https://aclanthology.org/2021.acl-long.274/)
- Nominal SRL
  - [Unsupervised Transfer of Semantic Role Models from Verbal to Nominal Domain](https://arxiv.org/pdf/2005.00278.pdf)

## 08/15/2021
- SLURM batch scripts tutorial
  - [video link](https://youtu.be/LRJMQO7Ercw)
  - <img width="1792" alt="Screen Shot 2021-08-15 at 9 23 07 PM" src="https://user-images.githubusercontent.com/32129905/129499719-412013eb-cbdc-4f77-b271-426e6d13f543.png">
  - <img width="1792" alt="Screen Shot 2021-08-15 at 9 22 54 PM" src="https://user-images.githubusercontent.com/32129905/129499727-151061aa-15a9-42d5-8462-a9872332c1a4.png">
  - <img width="1792" alt="Screen Shot 2021-08-15 at 9 22 28 PM" src="https://user-images.githubusercontent.com/32129905/129499744-af9249fd-d3fd-4e16-83cf-bee8536b470f.png">




## 08/11/2021
- GPT-3 usage
  - [link](https://www.reddit.com/r/GPT3/comments/kkp2py/17_examples_of_completions_from_the_new_gpt3/)
- Attention Mechanism
  - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- TPU
  - [RoBERTa meets TPUs](https://yassinealouini.medium.com/roberta-meets-tpus-af839ce7c070)
- Difference between BERT & RoBERTa
  - [BPE](https://medium.com/@pierre_guillou/byte-level-bpe-an-universal-tokenizer-but-aff932332ffe)
  - As you have seen above, RoBERTa uses a different tokenizer than the one used by BERT: byte-level BPE vs WordPiece. Here are the main differences between the two:
    - RoBERTa’s default tokenizer works at the byte-level vs word pieces for BERT.
    - RoBERTa’s tokenizer keeps all combined tokens (up to the vocabulary max size) whereas BERT’s only keeps those that increase 
- Acronym
  - tl;dr
  - ![1101611_1_0501-tldr-inaword_standard](https://user-images.githubusercontent.com/32129905/129098144-dd423188-d0bc-4ae4-81cd-32da17fc11ed.jpg)


## 07/28/2021
- GPT-3 size
  - Ada 350M
  - Babbage 1.3B
  - Curie 6.7B
  - Davinci 175B
  - [Reference](https://blog.eleuther.ai/gpt3-model-sizes/)


## 07/27/2021
- RoBERTa
  - cls ([CLS] in BERT), ```'<s>'```, 0
  - pad, ```'<pad>'```, 1
  - sep ([SEP] in BERT), ```'</s>'```, 2
  - unk, ```'<unk>'```, 3
  - [Reference](https://huggingface.co/transformers/_modules/transformers/tokenization_roberta.html)
- PairedRL
  - CLS, sentence1, SEP, sentence2, SEP
  - HiEve max: 155
  - IC max: 193
- BigBird
  - [BigBird](https://huggingface.co/transformers/model_doc/bigbird.html), is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and random attention approximates full attention, while being computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context, BigBird has shown improved performance on various long document NLP tasks, such as question answering and summarization, compared to BERT or RoBERTa.
  


## 07/26/2021
- [Shell字符串详解](http://c.biancheng.net/view/821.html)
  - <img width="1017" alt="Screen Shot 2021-07-26 at 4 40 03 PM" src="https://user-images.githubusercontent.com/32129905/127056053-65f10233-443b-4b8f-8137-a31b31301987.png">


## 07/24/2021
- [Shell字符串拼接（连接、合并）](http://c.biancheng.net/view/1114.html)
  - <img width="631" alt="Screen Shot 2021-07-26 at 4 38 06 PM" src="https://user-images.githubusercontent.com/32129905/127055835-7de4cbc3-a29c-4e2d-9b21-fa909d0f4705.png">
- [一篇教会你写90%的shell脚本](https://zhuanlan.zhihu.com/p/264346586)
- [curl shell](https://github.com/why2011btv/Quizlet_6/blob/master/te_out/my.sh)

## 07/23/2021
- Why does "source venv/bin/activate" not work?
  - <img width="1178" alt="Screen Shot 2021-07-23 at 12 52 02 AM" src="https://user-images.githubusercontent.com/32129905/126738758-6257a0b5-00b9-4fbf-990c-be1bc10468bf.png">
  - Tentative solution ([reference](https://www.devdungeon.com/content/python-import-syspath-and-pythonpath-tutorial#toc-5)): 
    - ```export VIRTUAL_ENV=/shared/public/ben/temporalextraction/venv```
    - ```export PATH="$VIRTUAL_ENV/bin:$PATH"```
    - ```export PYTHONHOME=/shared/public/ben/temporalextraction/venv```
    - ```export PYTHONPATH=/shared/public/ben/temporalextraction/venv/lib/python3.6```
  - [A similar problem](https://github.com/ContinuumIO/anaconda-issues/issues/172)
  - [Is my virtual environment (python) causing my PYTHONPATH to break?](https://stackoverflow.com/questions/4686463/is-my-virtual-environment-python-causing-my-pythonpath-to-break)
  - [How do you set your pythonpath in an already-created virtualenv?](https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv)
  - ![Screen Shot 2021-07-24 at 12 09 50 AM](https://user-images.githubusercontent.com/32129905/126856867-07252935-5240-47ed-952d-648b6c9505da.png)
  - [python编程【环境篇】- 如何优雅的管理python的版本](https://www.cnblogs.com/zhouliweiblog/p/11497045.html)
  - [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [Quizlet 6 Draft Pipeline](https://github.com/why2011btv/Quizlet_6)



<html lang="en-US">
    <meta charset="UTF-8">
<body>

<h1>MYLOG_old</h1>
<h2>07/23/2020</h2>
<h3>conda activate /shared/why16gzl/logic_driven/consistency_env</h3>
<h3>conda activate myenv</h3>
<h3>source ~/venv/bin/activate</h3>
<h3>source /shared/why16gzl/logic_driven/mandarjoshi90_coref/mandarjoshi90_coref_env/bin/activate</h3>
<li><p><a href="https://www.w3schools.com/python/gloss_python_global_variables.asp">Python Global Variables</a>: "global x" inside a function does not define a new variable</p></li>
<li><p><a href="https://www.geeksforgeeks.org/multiprocessing-python-set-2/">Multiprocessing in Python</a>: any newly created process will have their own memory space</p></li>
<li><p><a href="https://towardsdatascience.com/virtual-environments-104c62d48c54">Creating a Virtual Environment</a>:<br>mkdir new_project<br>cd new_project<br>python3 -m venv venv&#60;name_of_virtualenv&#62;<br>source venv/bin/activate<br>deactivate<br>pip freeze &#62; requirements.txt</p></li>
<li><p><a href="https://realpython.com/python-virtual-environments-a-primer/">Distinguishing: virtualenv, virtualenvwrapper, and pyenv</a></p></li>
<li><p><a href="https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533">Distinguishing: conda, pip, and venv</a>:<br>conda create --prefix /path/to/conda-env<br>conda create --name conda-env python #installed in /home1/w/why16gzl/miniconda3/envs/</p></li>
Given an environment.yml file, you can easily recreate an environment.<br>
% conda env create -n conda-env -f /path/to/environment.yml<br>
Bonus: You can also add the packages listed in an environment.yml file to an existing environment with:<br>
% conda env update -n conda-env -f /path/to/environment.yml
<li><p><a href="https://realpython.com/python-virtual-environments-a-primer/">virtualenvwrapper</a></p></li>
<h2>07/24/2020</h2>
<h3>vi line navigation</h3>
0: go to start of line<br>
$: go to end of line<br>
1G: go to top of file<br>
5G: go to line 5<br>
G: go to end of file<br>
/: search for string<br>
    
    
    
<h2>07/25/2020</h2>
<h3>set up environment for <a href="https://github.com/mandarjoshi90/coref/">SpanBERT coref part</a></h3>
pip install cort:<br>git clone https://github.com/smartschat/cort.git<br>go to setup.py line 37, delete "mmh3"<br>conda install mmh3<br>python setup.py install
<h3><a href="https://linuxize.com/post/how-to-create-symbolic-links-in-linux-using-the-ln-command/">Create Symbolic Links</a></h3>
ln -s /shared/why16gzl/miniconda3/myenv/ /home1/w/why16gzl/miniconda3/envs/myenv<br>
conda activate myenv (or consistency_env, coref_SpanBERT_TF_env, directly without specifying whole path)
<h3>ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory</h3>
See /home1/w/why16gzl/0_markdown/TF_ipykernel_Configuration.ipynb<br>
Now myenv has tensorflow==1.12.0
<h3>List the size of directory: du -h --max-depth=1 | sort -n</h3>
<h2>07/26/2020</h2>
<h3>Improving Implicit Argument and Explicit Event Connections for Script Event Prediction, EMNLP 2020 Submission</h3>
To capture explicit evolutionary relationships between events, we follow (Li et al., 2018) to construct an event graph from intersecting chains, and further develop graph convolutional layers based on GCN (Kipf and Welling, 2017) to extract fine-grained features.
<h3>Constructing Narrative Event Evolutionary Graph for Script Event Prediction, IJCAI 2018</h3>
In order to compare with previous work, we adopt the same news corpus and event chains extraction methods as [Granroth-Wilding and Clark, 2016].
<h3>What Happens Next? Event Prediction Using a Compositional Neural Network Model, AAAI 2016</h3>
This paper follows a line of work begun by Chambers and Jurafsky (2008), who introduced a technique for automatically extracting knowledge about typical sequences of events from text.<br>
we propose a new task, multiple choice narrative cloze (MCNC)<br>
The event extraction pipeline follows an almost identical procedure to Chambers and Jurafsky (2009), using the C&amp;C tools (Curran, Clark, and Bos 2007) for PoS tagging and dependency parsing and OpenNLP for phrase-structure parsing and coreference resolution.
<h3>Unsupervised learning of narrative event chains, ACL 2008</h3>
Narrative Cloze Task
<h2>07/27/2020</h2>
<h3>How to run OneIE</h3>
conda activate consistency_env<br>
python predict.py -m ../english.role.v0.3.mdl -i input -o output -c output_cs --format ltf
<h3>How to run SpanBERT</h3>
conda activate coref_SpanBERT_TF_env<br>
CUDA_VISIBLE_DEVICES=0 python predict.py spanbert_large cased_config_vocab/trial.jsonlines 0725.out
<h3>How to build the event graph</h3>
First run OneIE to get: segment-15 ... [3, 5, "GPE", "NAM", 1.0] ... [9, 10, "Contact:Meet", 1.0] ... [1, 0, "ORG-AFF", 0.52] ... [0, 2, "Attacker", 0.5514472874407822]. Represent each entity/event as their #segment_#start_token_index_#end_token_index: 15_3_5
<br>
Then run SpanBERT. SpanBERT input can be several sentences. But the token num is limited.
<h2>08/05/2020</h2>
<h3>How to debug</h3>
<h3><a href="https://www.codementor.io/@stevek/advanced-python-debugging-with-pdb-g56gvmpfa">Post-mortem debugging</a></h3>
python3 -mpdb script.py<br>
To start execution, you use the continue or c command. If the program executes successfully, you will be taken back to the (Pdb) prompt where you can restart the execution again. If the program throws an unhandled exception, you'll also see a (Pdb) prompt, but with the program execution stopped at the line that threw the exception. From here, you can run Python code and debugger commands at the prompt to inspect the current program state.
<h3><a href="https://realpython.com/lessons/continuing-execution/">A tutorial on pdb</a></h3>
<h3><a href="http://qingkaikong.blogspot.com/2018/05/python-debug-in-jupyter-notebook.html">Python debug in Jupyter notebook</a></h3>
%debug
<h3><a href="https://davidhamann.de/2017/04/22/debugging-jupyter-notebooks/">Debugging Jupyter notebooks</a></h3>
<h3><a href="https://stackoverflow.com/questions/2534480/proper-way-to-reload-a-python-module-from-the-console">Proper way to reload a python module from the console</a></h3>
import document_reader<br>
import importlib<br>
importlib.reload(document_reader)<br>
from document_reader import *<br>
<h3><a href="https://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing">Unsolved Problem: nltk.tokenizer would change "" to ``''</a></h3>
<h3><a href="https://mccormickml.com/2019/07/22/BERT-fine-tuning/#4-train-our-classification-model">BERT Fine-Tuning Tutorial with PyTorch</a></h3>
<h3><a href="">xxx</a></h3>
<h2>08/16/2020</h2>
<h3>Install CUDA 10.2</h3>
1) go to <a href="https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=OpenSUSE&target_version=15&target_type=runfilelocal">nvidia website</a><br>
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run<br>
chmod a+x cuda_10.2.89_440.33.01_linux.run<br>
sh ./cuda_10.2.89_440.33.01_linux.run<br>
2) change the options (installation path) <a href="https://stackoverflow.com/questions/39379792/install-cuda-without-root">see answer by fr_andres</a><br>
3) add below to .bashrc<br>
#cuda-10.2<br>
export PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/bin:$PATH<br>
export LD_LIBRARY_PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/lib64<br>
<h3><a href="https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory">How do I list all files of a directory</a></h3>
<h3><a href="https://github.com/thunlp/DocRED">Dataset and code for baselines for DocRED: A Large-Scale Document-Level Relation Extraction Dataset</a></h3>
<a href="https://www.aclweb.org/anthology/P08-1090.pdf">Unsupervised Learning of Narrative Event Chains</a>
<h3><a href="http://qingkaikong.blogspot.com/2018/05/python-debug-in-jupyter-notebook.html"> Python debug in Jupyter notebook </a></h3>
<h2>08/17/2020</h2>
<h3>torch.stack</h3>
In [ ]:<br>
a = torch.tensor([0, 1, 2])<br>
b = []<br>
b.append(a)<br>
b.append(a)<br>
torch.stack(b)<br>
Out [ ]:<br>
tensor([[0, 1, 2],<br>
        [0, 1, 2]])<br>
<h3>RoBERTa token id</h3>
&#60;s&#62;: 0<br>
&#60;pad&#62;: 1<br>
&#60;/s&#62;: 2<br>
&#60;unk&#62;: 3<br>
<h3><a href="https://www.teachucomp.com/special-characters-in-html-tutorial/">special characters in html</a></h3>
<h3>How to run my model (entity coref as context)</h3>
/shared/why16gzl/logic_driven/Untitled1.ipynb<br>
coref_flag = True: train_set_coref is generated<br>
coref_flag = False: train_set is generated<br>
Under folder /shared/why16gzl/logic_driven/data_pkl
<h2>08/20/2020</h2>
<h3>Update github repo</h3>
git add .<br>
git commit -m "add googlescholar"<br>
git config http.postBuffer 524288000 # if the next line does not work you might try this line first<br>
git push origin master<br>
<h3><a href="https://www.aclweb.org/anthology/W18-4702.pdf">Interoperable Annotation of Events and Event Relations across Domains</a></h3>
<h2>08/25/2020</h2>
<h3>Not using pickle, training after dataloader</h3>
(consistency_env) why16gzl@morrison:/shared/why16gzl/logic_driven/EMNLP_2020&#62; nohup python3 main_0824.py gpu_3 batch_28 0.00001 0824_1.rst epoch_60 &#62; 0824_1.out 2&#62;&#38;1 &#38;<br>
<h3>Using pickle, coref or no_coref</h3>
<h4>Preprocessing, Coref</h4>
(allennlp_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; python why16gzl_prep.py 1
<h4>Preprocessing, No Coref</h4>
(allennlp_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; python why16gzl_prep.py 0
<h4>Training</h4>
(consistency_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; nohup python3 main.py gpu_1 batch_24 no_coref 0817_0.rst epoch_60 &#62; 0817_0.out 2&#62;&#38;1 &#38;<br>
<h2>09/05/2020</h2>
<h3>CogCompNLP environment</h3>
A JAR (Java ARchive) is a package file format typically used to aggregate many Java class files and associated metadata and resources (text, images, etc.) into one file for distribution. JAR files are archive files that include a Java-specific manifest file.<br>
Maven is a build automation tool used primarily for Java projects. Maven can also be used to build and manage projects written in C#, Ruby, Scala, and other languages. The Maven project is hosted by the Apache Software Foundation, where it was formerly part of the Jakarta Project.<br>
<h2>09/20/2020</h2>
Create a new repository on GitHub. You can also add a gitignore file, a readme and a licence if you want<br>
 Open Git Bash<br>
Change the current working directory to your local project.<br>
Initialize the local directory as a Git repository.<br>
git init<br>
Add the files in your new local repository. This stages them for the first commit.<br>
git add .<br>
 Commit the files that you’ve staged in your local repository.<br>
git commit -m "initial commit"<br>
 Copy the https url of your newly created repo<br>
In the Command prompt, add the URL for the remote repository where your local repository will be pushed.<br>

git remote add origin remote repository URL<br>

git remote -v<br>
 Push the changes in your local repository to GitHub.<br>

git push -f origin master<br>
https://www.softwarelab.it/2018/10/12/adding-an-existing-project-to-github-using-the-command-line/<br>
https://stackoverflow.com/questions/15244644/how-to-restart-a-git-repository<br>
https://stackoverflow.com/questions/32238616/git-push-fatal-origin-does-not-appear-to-be-a-git-repository-fatal-could-n<br>
<h2>09/22/2020</h2>
https://shmsw25.github.io<br>
https://seominjoon.github.io<br>
https://luheng.github.io/files/cv_luheng_2017.pdf<br>

HW1:<br>
Part 1: Querying an AWS Oracle database (IMDB database)<br>
Use "Oracle SQL Developer" to connect to IMDB database<br>
Part 2: Setting up and querying an OpenFlights dataset using MySQL (MariaDB)<br>
<h2>09/28/2020</h2>
"pip install en-core-web-sm==2.1.0" fails, instead, do the following:<br>
python -m spacy download en_core_web_sm<br>

Go to end of file (vi): &#60;ESC&#62;GA

<h2>10/04/2020</h2>
linux 下后台运行python脚本
<br>
https://www.jianshu.com/p/4041c4e6e1b0


<h2>01/09/2021</h2>

$ git tag -a v1.4 -m "my version 1.4"
<br>
https://git-scm.com/book/en/v2/Git-Basics-Tagging
<br>
git add . <br>
git commit -m "xxx" <br>
git push origin v1.4
<br>

<h2>01/10/2021</h2>
multiprocessing
<br>
https://pymotw.com/2/multiprocessing/basics.html
<br>
threading
<br>
https://pymotw.com/2/threading/index.html#module-threading
<br>

<h2>01/18/2021</h2>
git checkout -b <branch> <br>
Edit files, add and commit. Then push with the -u (short for --set-upstream) option:
<br>
git push -u origin <branch>
<br>

https://stackoverflow.com/questions/2765421/how-do-i-push-a-new-local-branch-to-a-remote-git-repository-and-track-it-too

<h2>03/19/2021</h2>
https://www.bradyneal.com/which-causal-inference-book#elements-of-causal-inference<br>
<h2>03/21/2021</h2>

import notify<br>
see: https://github.com/why2011btv/JCL_EMNLP20/blob/main/notify_message.py and https://github.com/why2011btv/JCL_EMNLP20/blob/main/notify_smtp.py and https://github.com/why2011btv/JCL_EMNLP20/blob/0ede3a5c205c57b6e82499ad5ed46f2af63acafc/exp.py#L250<br>



shift+G: go to end of file (vi)<br>
How to remove files already added to git after you update .gitignore: git rm -r --cached .<br>
https://itnext.io/how-to-remove-files-already-added-to-git-after-you-update-gitignore-90f169a0a4e1<br>
https://www.javatpoint.com/git-head<br>
To delete the last line of file: sed -i '$d' FILENAME<br>
git ls-files<br>

https://opensource.com/article/18/6/git-reset-revert-rebase-commands<br>
$ git log --oneline<br>
b764644 File with three lines<br>
7c709f0 File with two lines<br>
9ef9173 File with one line<br>

$ git reset 9ef9173 (using an absolute commit SHA1 value 9ef9173)<br>
<h2>03/22/2021</h2>
http://www.filepermissions.com/directory-permission/2750<br>
<h2>03/31/2021</h2>
How to get the root node: in-degree == 0<br>
https://stackoverflow.com/questions/4122390/getting-the-root-head-of-a-digraph-in-networkx-python<br>

<h2>04/24/2021</h2>
git push to github, not showing user picture in commit info, but only "Haoyu Wang": <br>
https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address<br>
https://stackoverflow.com/questions/48816950/showing-gitlab-profile-picture-on-commits<br>
You need to: git config user.email "your email here"<br>
https://docs.github.com/en/github/setting-up-and-managing-your-github-profile/why-are-my-contributions-not-showing-up-on-my-profile#commit-was-made-less-than-24-hours-ago<br>
Why sometimes it does not count towards your contribution?


</body>
</html>
    
## 07/12/2021
<img width="1361" alt="Screen Shot 2021-07-12 at 11 56 23 PM" src="https://user-images.githubusercontent.com/32129905/125388101-e30a2c80-e36c-11eb-84ad-8ac0053ba2fc.png">
    
## 07/13/2021
pip list
    

