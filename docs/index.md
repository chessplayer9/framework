# Introduction

The OpenHAIV framework primarily focuses on deep learning-based computer vision tasks and currently supports the following:

## Supervised Learning
Supervised learning is one of the most commonly used machine learning algorithms. In the field of computer vision, tasks such as image classification typically rely on labeled data for training. Labeled data refers to paired input data and corresponding target outputs (labels). Through these pairs, the model learns the mapping from input to output. During training, the model continuously adjusts its internal parameters to minimize the gap between predictions and actual labels. A fundamental training framework should support supervised learning, and Openhaiv allows the use of different deep learning models, datasets, and flexible hyperparameter tuning.

## Incremental Learning
Incremental learning is an advanced machine learning paradigm designed to address the issue of catastrophic forgetting in machine learning. Specifically, incremental learning enables models to learn new knowledge while retaining and optimizing previously acquired knowledge. Currently, mainstream incremental learning methods can be categorized into three types: regularization-based methods, experience replay-based methods, and parameter isolation-based methods. The Openhaiv framework currently supports multiple incremental learning algorithms, including ALICE, FACT, and SAVC, allowing users to select and adjust them based on task requirements.

## Few-Shot Learning
Few-shot learning is a common scenario in deep learning, where models must learn and make inferences effectively with only a small number of samples. Specifically, few-shot learning uses a limited number of samples during training (typically 5–10 samples per class). The OpenHAIV framework currently supports few-shot learning, allowing users to define models, datasets, and the number of samples per class in the few-shot stage. Few-shot learning often employs methods such as contrastive learning and augmentation techniques, which are well-supported in Openhaiv. Additionally, the framework is highly extensible, enabling the integration of new few-shot learning methods.

## Out-of-Distribution Detection
Out-of-distribution detection is the process of identifying whether data deviates from the known data distribution. On one hand, OOD detection ensures that models can make reliable judgments when encountering inputs different from known classes, thereby improving system robustness. On the other hand, OOD detection is a crucial component of incremental object recognition in open environments. The Openhaiv framework currently supports multiple OOD detection algorithms, allowing flexible adjustments for different domains and tasks.

## Novel Category Discovery
Novel category discovery refers to identifying and discovering new categories in data without pre-existing labeled classes. Currently, novel category discovery typically employs unsupervised clustering algorithms in machine learning. Unsupervised clustering groups datasets into clusters, and Openhaiv supports the use of the K-Means clustering algorithm for novel category discovery. K-Means partitions data into K clusters, with each cluster represented by its centroid (the center point of the cluster). The algorithm iteratively updates centroids to minimize the within-cluster squared distances.
