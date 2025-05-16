## Support Methods for OpenHAIV

### Class-incremental learning

- `Joint`: update models using all the data from all classes.
- `Finetune`: baseline method which simply update model using current data.
- `LwF`: Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
-  `EWC`: Overcoming catastrophic forgetting in neural networks. PNAS 2017 [[paper](https://arxiv.org/abs/1612.00796)]
-  `iCaRL`: Incremental Classifier and Representation Learning. CVPR 2017 [[paper](https://arxiv.org/abs/1611.07725)]
-  `BiC`: Large Scale Incremental Learning. CVPR 2019 [[paper](https://arxiv.org/abs/1905.13260)]
-  `WA`: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR 2020 [[paper](https://arxiv.org/abs/1911.07053)]
-  `DER`: DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR 2021 [[paper](https://arxiv.org/abs/2103.16788)]
-  `Coil`: Co-Transport for Class-Incremental Learning. ACM MM 2021 [[paper](https://arxiv.org/abs/2107.12654)]
- `FOSTER`: Feature Boosting and Compression for Class-incremental Learning. ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]
-  `SSRE`: Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning. CVPR2022 [[paper](https://arxiv.org/abs/2203.06359)]
- `FeTrIL`: Feature Translation for Exemplar-Free Class-Incremental Learning. WACV2023 [[paper](https://arxiv.org/abs/2211.13131)]
-  `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. ICLR 2023 Spotlight [[paper](https://openreview.net/forum?id=S07feAlQHgM)]

### Out-of-Distribution Detection
#### Unimodal Methods
- `MSP`: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017[[paper](https://arxiv.org/abs/1610.02136)]
- `ODIN`: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. ICLR 2018[[paper](https://arxiv.org/abs/1706.02690)]
- `MDS`: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018[[paper](https://arxiv.org/abs/1807.03888)]
- `GRAM`: Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices. ICML 2020[[paper](https://arxiv.org/abs/1912.12510)]
- `EBO`: Energy-based Out-of-distribution Detection. NeurIPS 2020[[paper](https://arxiv.org/abs/2010.03759)]
- `RMDS`: A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. Arxiv 2021[[paper](https://arxiv.org/abs/2106.09022)]
- `GranNorm`: On the Importance of Gradients for Detecting Distributional Shifts in the Wild. NeurIPS 2021[[paper](https://arxiv.org/abs/2110.00218)]
- `React`: ReAct: Out-of-distribution Detection With Rectified Activations. NeurIPS 2021[[paper](https://arxiv.org/abs/2111.12797)]
- `SEM`: . Arxiv 2022[[paper]()]
- `MLS`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KLM`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KNN`: Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022[[paper](https://arxiv.org/abs/2204.06507)]
- `vim`: ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022[[paper](https://arxiv.org/abs/2203.10807)]
- `Dice`: DICE: Leveraging Sparsification for Out-of-Distribution Detection. ECCV 2022[[paper](https://arxiv.org/abs/2111.09805)]
- `RankFeat`: RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection. NeurIPS 2022[[paper](https://arxiv.org/abs/2209.08590)]
- `ASH`: Extremely Simple Activation Shaping for Out-of-Distribution Detection. ICLR 2023[[paper](https://arxiv.org/abs/2209.09858)]
- `SHE`: . ICLR 2023[[paper]()]
- `GEN`:  GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]
- `NNGuide`: Nearest Neighbor Guidance for Out-of-Distribution Detection. ICCV 2023[[paper](https://arxiv.org/abs/2309.14888)]
- `Relation`: Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data. NeurIPS 2023[[paper](https://arxiv.org/abs/2301.12321)]
- `Scale`: Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement. ICLR 2024[[paper](https://arxiv.org/abs/2310.00227)]
- `FDBD`: Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024[[paper](https://arxiv.org/abs/2312.11536)]
<!--- `AdaScale A`: AdaSCALE: Adaptive Scaling for OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2503.08023)]-->
<!--- `AdaScale L`: AdaSCALE: Adaptive Scaling for OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2503.08023)]-->
<!--- `IODIN`: Going Beyond Conventional OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2411.10794)]-->
<!--- `NCI`: Detecting Out-of-Distribution Through the Lens of Neural Collapse. CVPR 2025[[paper](https://arxiv.org/abs/2311.01479)]-->
#### CLIP-based Methods
- `MCM`: Delving into Out-of-Distribution Detection with Vision-Language Representations. NeurIPS 2022[[paper](https://arxiv.org/abs/2211.13445)]
- `NegLabel`: Negative Label Guided OOD Detection with Pretrained Vision-Language Models. ICLR 2024[[paper](https://arxiv.org/abs/2403.20078)]
- `CoOp`: Learning to Prompt for Vision-Language Models. IJCV 2022[[paper](https://arxiv.org/abs/2109.01134)]
- `LoCoOp`: LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning. NeurIPS 2023[[paper](https://arxiv.org/abs/2306.01293)]
- `SCT`: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection. NeurIPS 2024[[paper](https://arxiv.org/abs/2411.03359)]
- `Maple`: MaPLe: Multi-modal Prompt Learning. CVPR 2023[[paper](https://arxiv.org/abs/2210.03117)]
- `DPM`: Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection. ECCV 2024[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11399.pdf)]
- `Tip-Adapter`: Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. ECCV 2022[[paper](https://arxiv.org/abs/2111.03930)]
- `NegPrompt`: Learning Transferable Negative Prompts for Out-of-Distribution Detection. CVPR 2024[[paper](https://arxiv.org/abs/2404.03248)]

- `GL-MCM`: GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection. IJCV 2025 [[paper](https://arxiv.org/abs/2304.04521)]

### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV 2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning. CVPR 2023 [[paper](https://arxiv.org/abs/2304.00426)]
