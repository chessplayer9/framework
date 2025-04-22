# ncdia

## ncdia.algorithms
Includes incremental learning algorithms, new class discovery algorithms, and out-of-distribution detection algorithms, as well as supervised learning algorithms.

### ncdia.algorithms.base.py

#### BaseAlg
Basic algorithm class to define the interface of an algorithm.

- <span class="highlight-text">**\_\_init\_\_(self, trainer)**</span>

    The constructor method that initializes an instance of **BaseAlg**.

    **Parameters:**

    - **trainer** (*object*): Trainer object.

- <span class="highlight-text">**train_step(self, trainer, data, label, *args, **kwargs)**</span>

    Training step.

    **Parameters:**

    - **trainer** (*object*): Trainer object.
    - **data** (*torch.Tensor*): Input data.
    - **label** (*torch.Tensor*): Label data.
    - **args** (*tuple*): Additional arguments.
    - **kwargs** (*dict*): Additional keyword arguments.

    **Returns:**

    - **results** (*dict*): Training results. Contains the following keys:
        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.
        - **other** "key:value" pairs.

- <span class="highlight-text">**val_step(self, trainer, data, label, *args, **kwargs)**</span>

    Validation step.

    **Parameters:**

    - **trainer** (*object*): Trainer object.
    - **data** (*torch.Tensor*): Input data.
    - **label** (*torch.Tensor*): Label data.
    - **args** (*tuple*): Additional arguments.
    - **kwargs** (*dict*): Additional keyword arguments.

    **Returns:**

    - **results** (*dict*): Validation results. Contains the following keys:
        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.
        - **other** "key:value" pairs.

- <span class="highlight-text">**test_step(self, trainer, data, label, *args, **kwargs)**</span>

    Test step.

    **Parameters:**

    - **trainer** (*object*): Trainer object.
    - **data** (*torch.Tensor*): Input data.
    - **label** (*torch.Tensor*): Label data.
    - **args** (*tuple*): Additional arguments.
    - **kwargs** (*dict*): Additional keyword arguments.

    **Returns:**

    - **results** (*dict*): Test results. Contains the following keys:
        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.
        - **other** "key:value" pairs.

### ncdia.algorithms.incremental

Include implementation of Class Incremental Learning (CIL) and Few-shot Class Incremental Learning (FSCIL) algorithm.

- **CIL**
    - **Finetune**
    - **LwF**
    - **EwC**
    - **iCarL**
    - **IL2A**
    - **WA**    
- **FSCIL**
    - **ALICE**
    - **FACT**
    - **SAVC**

### ncdia.algorithms.ncd.autoncd.py
Modules related to novel class discovery.

#### AutoNCD
Class for evaluating with OOD metrics and relabeling the OOD dataset for the next session.

- <span class="highlight-text">**\_\_init\_\_(self, model, train_loader, test_loader, device=None, verbose=False)**</span>

    The constructor method that initializes an instance of **AutoNCD**. 

    **Parameters:**

    - **model** (*nn.Module*): model to be evaluated.
    - **train_loader** (*DataLoader*): train dataloader.
    - **test_loader** (*DataLoader*): test dataloader.
    - **device** (*torch.device, optional*): device to run the evaluation. Default to None.
    - **verbose** (*bool, optional*): print the progress bar. Default to False. 

- <span class="highlight-text">**inference(self, dataloader, split='train')**</span>

    Inference the model on the dataloader and return relevant information. If split is 'train', return the prototype of the training data. 

    **Parameters:**

    - **dataloader** (*DataLoader*): dataloader for evaluation.
    - **split** (*str, optional*): train or test. Defaults to 'train'.

    **Returns:**

    If split is 'train':

    - **features** (*torch.Tensor*): feature vectors, (N, D).

    - **logits** (*torch.Tensor*): logit vectors, (N, C).

    - **prototype_cls** (*torch.Tensor*): prototype vectors, (C, D).

    If split is 'test':

    - **imgpaths** (*list*): image paths (list).

    - **features** (*torch.Tensor*): feature vectors, (N, D).

    - **logits** (*torch.Tensor*): logit vectors, (N, C).

    - **preds** (*torch.Tensor*): prediction labels, (N,).

    - **labels** (*torch.Tensor*): ground truth labels, (N,).

- <span class="highlight-text">**relabel(self, ood_loader, metrics=[], tpr_th=0.95, prec_th=None)**</span>

    Relabel the OOD dataset for the next session.

    **Parameters:**

    - **ood_loader** (*DataLoader*): OOD dataloader for relabeling.
    - **metrics** (*list, optional*): metrics to evaluate the OOD dataset. Defaults to [].
    - **tpr_th** (*float, optional*): True positive rate threshold. Defaults to 0.95.
    - **prec_th** (*float, optional*): Precision threshold. Defaults to None.

    **Returns:**

    - **ood_loader** (*DataLoader*): relabeled OOD dataloader.

- <span class="highlight-text">**_split_cluster_label(self, y_label, y_pred, ood_class)**</span>

    Calculate clustering accuracy. Require scikit-learn installed. First compute linear assignment on all data, then look at how good the accuracy is on subsets.

    **Parameters:**

    - **y_label** (*numpy.array*): true labels, (n_samples,)
    - **y_pred** (*numpy.array*): predicted labels (n_samples,)
    - **ood_class**: out-of-distribution class labels

    **Returns:**

    - **cluster_label**: cluster label

- <span class="highlight-text">**search_discrete_point(self, novel_feat, novel_target)**</span>

<span style="color: red;">**TODO**</span>

### ncdia.algorithms.ood.autoood.py

#### AutoOOD

Class for evaluating OOD detection methods.

- <span class="highlight-text">**eval(prototype_cls, fc_weight, train_feats, train_logits, id_feats, id_logits, id_labels, ood_feats, ood_logits, ood_labels, metrics=[], tpr_th=0.95, prec_th=None, id_attrs=None, ood_attrs=None, prototype_att=None)**</span>

    Evaluate the OOD detection methods and return OOD scores.

    **Parameters:**

    - **prototype_cls** (*np.ndarray*): prototype of training data
    - **fc_weight** (*np.ndarray*): weight of the last layer
    - **train_feats** (*np.ndarray*): feature of training data
    - **train_logits** (*np.ndarray*): logits of training data
    - **id_feats** (*np.ndarray*): feature of ID data
    - **id_logits** (*np.ndarray*): logits of ID data
    - **id_labels** (*np.ndarray*): labels of ID data
    - **ood_feats** (*np.ndarray*): feature of OOD data
    - **ood_logits** (*np.ndarray*): logits of OOD data
    - **ood_labels** (*np.ndarray*): labels of OOD data
    - **metrics** (*list, optional*): list of OOD detection methods to evaluate. Defaults to [].
    - **tpr_th** (*float, optional*): True positive rate threshold. Defaults to 0.95.
    - **prec_th** (*float, optional*): Precision threshold. Defaults to None.

    **Returns:**

    - **ood_scores** (*dict*): OOD scores, keys are the names of the OOD detection methods, values are the OOD scores and search threshold. Each value is a tuple containing the following:
        - **ood metrics** (*tuple*):
            - **fpr** (*float*): false positive rate
            - **auroc** (*float*): area under the ROC curve
            - **aupr_in** (*float*): area under the precision-recall curve for in-distribution samples
            - **aupr_out** (*float*): area under the precision-recall curve for out-of-distribution samples
        - **search threshold** (*tuple*): threshold for OOD detection if prec_th is not None
            - **best_th** (*float*): best threshold for OOD detection
            - **conf** (*torch.Tensor*): confidence scores
            - **label** (*torch.Tensor*): label array
            - **precisions** (*float*): precision when precisions >= prec_th
            - **recalls** (*float*): recall when precisions >= prec_th

- <span class="highlight-text">**inference(metrics, logits, feat, train_logits, train_feat, fc_weight, prototype, logits_att=None, prototype_att=None)**</span>

    Inferencec method for OOD detection

    **Parameters:**

    - **metrics** (*list*): the ood metrics used for inference.
    - **logits** (*np.ndarray*): logits of inference data.
    - **feat** (*np.ndarray*): features of inference data.
    - **train_logits** (*np.ndarray*): logits of training data.
    - **train_feat** (*np.ndarray*): features of training data.
    - **fc_weight** (*np.ndarray*): weight of the last layer.
    - **prototype** (*np.ndarray*): prototypes of training data.
    - **logits_att** (*np.ndarray, optional*): logits of attribute.
    - **prototype_att** (*np.ndarray, optional*): prototypes of attribute. 

    **Returns:**

    - **conf** (*dict*): contains the confidence using different metrics, **conf[metric]** (*torch.Tensor*) is the confidence using specific metric.

### ncdia.algorithms.ood.methods.py

- <span class="highlight-text">**msp(id_gt, id_logits, ood_gt, ood_logits, tpr_th=0.95, prec_th=None)**</span>

    Maximum Softmax Probability (MSP) method for OOD detection.

    A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching - threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **conf** (*np.ndarray*): Confidence scores. Shape (N + M,).
    - **label** (*np.ndarray*): Label array. Shape (N + M,).
    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution
    - **best_th** (*float*): Threshold for OOD detection. If prec_th is None, None.
    - **prec** (*float*): Precision at the threshold. If prec_th is None, None.
    - **recall** (*float*): Recall at the threshold. If prec_th is None, None.

- <span class="highlight-text">**mcm(id_gt, id_logits, ood_gt, ood_logits, T=2, tpr_th=0.95, prec_th=None)**</span>

    Maximum Concept Matching (MCM) method for OOD detection.

    Delving into Out-of-Distribution Detection with Vision-Language Representations

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **T** (*int*): Temperature for softmax.
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution.

- <span class="highlight-text">**max_logit(id_gt, id_logits, ood_gt, ood_logits, tpr_th=0.95, prec_th=None)**</span>

    Maximum Logit (MaxLogit) method for OOD detection.

    Scaling Out-of-Distribution Detection for Real-World Settings

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

- <span class="highlight-text">**energy(id_gt, id_logits, ood_gt, ood_logits, tpr_th=0.95, prec_th=None)**</span>

    Energy-based method for OOD detection.

    Energy-based Out-of-distribution Detection

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

- <span class="highlight-text">**vim(id_gt, id_logits, id_feat, ood_gt, ood_logits, ood_feat, train_logits, train_feat, tpr_th=0.95, prec_th=None)**</span>

    Virtual-Logit Matching (ViM) method for OOD detection.

    ViM: Out-of-Distribution With Virtual-Logit Matching

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **id_feat** (*torch.Tensor*): ID features. Shape (N, D).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **ood_feat** (*torch.Tensor*): OOD features. Shape (M, D).
    - **train_logits** (*torch.Tensor*): Training logits. Shape (K, C).
    - **train_feat** (*torch.Tensor*): Training features. Shape (K, D).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

- <span class="highlight-text">**dml(id_gt, id_feat, ood_gt, ood_feat, fc_weight, tpr_th=0.95, prec_th=None)**</span>

    Decoupled MaxLogit (DML) method for OOD detection.

    Decoupling MaxLogit for Out-of-Distribution Detection

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_feat** (*torch.Tensor*): ID features. Shape (N, D).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_feat** (*torch.Tensor*): OOD features. Shape (M, D).
    - **fc_weight** (*torch.Tensor*): FC layer weight. Shape (C, D).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

- <span class="highlight-text">**dmlp(id_gt, id_logits, id_feat, ood_gt, ood_logits, ood_feat, fc_weight, prototype,tpr_th=0.95, prec_th=None)**</span>

    Decoupled MaxLogit+ (DML+) method for OOD detection.

    Decoupling MaxLogit for Out-of-Distribution Detection

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels. Shape (N,).
    - **id_logits** (*torch.Tensor*): ID logits. Shape (N, C).
    - **id_feat** (*torch.Tensor*): ID features. Shape (N, D).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels. Shape (M,).
    - **ood_logits** (*torch.Tensor*): OOD logits. Shape (M, C).
    - **ood_feat** (*torch.Tensor*): OOD features. Shape (M, D).
    - **fc_weight** (*torch.Tensor*): FC layer weight. Shape (D, C).
    - **prototype** (*torch.Tensor*): Prototype. Shape (D, C).
    - **tpr_th** (*float*): True positive rate threshold to compute
        false positive rate. Default is 0.95.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is None.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

- <span class="highlight-text">**prot(id_gt, id_logits, ood_gt, ood_logits, prototypes: list, tpr_th=0.95, prec_th=None)**</span>

    Prototype-based (Prot) method for OOD detection.

    **Parameters:**

    - **id_gt** (*torch.Tensor*): ID ground truth labels, shape (N,).
    - **id_logits** (*list of torch.Tensor*): ID logits, containing shape (N, C).
    - **ood_gt** (*torch.Tensor*): OOD ground truth labels, shape (M,).
    - **ood_logits** (*list of torch.Tensor*): OOD logits, containing shape (M, C).
    - **prototypes** (*list of torch.Tensor*): Prototypes, containing shape (D, C).
    - **tpr_th** (*float*): True positive rate threshold to compute 
        false positive rate.
    - **prec_th** (*float*): Precision threshold for searching threshold.
        If None, not searching for threshold. Default is

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution

### ncdia.algorithms.ood.inference.py

The inference version of implemented ood methods in [ncdia.algorithms.ood.methods.py](#ncdiaalgorithmsoodmethodspy)

### ncdia.algorithms.ood.metrics.py

- <span class="highlight-text">**ood_metrics(conf, label, tpr_th=0.95)**</span>

    Compute OOD metrics.

    **Parameters:**

    - **conf** (*np.ndarray*): Confidence scores. Shape (N,).
    - **label** (*np.ndarray*): Label array. Shape (N,). Containing:
        -1: OOD samples.
        int >= 0: ID samples with class labels
    - **tpr_th** (*float*): True positive rate threshold to compute 
        false positive rate.

    **Returns:**

    - **fpr** (*float*): False positive rate.
    - **auroc** (*float*): Area under the ROC curve.
    - **aupr_in** (*float*): Area under the precision-recall curve 
        for in-distribution samples.
    - **aupr_out** (*float*): Area under the precision-recall curve
        for out-of-distribution samples.

- <span class="highlight-text">**search_threshold(conf, label, prec_th)**</span>

    Search for the threshold for OOD detection.

    **Parameters:**

    - **conf** (*np.ndarray*): Confidence scores. Shape (N,).
    - **label** (*np.ndarray*): Label array. Shape (N,). Containing:
        -1: OOD samples.
        int >= 0: ID samples with class labels
    - **prec_th** (*float*): Precision threshold.

    **Returns:**

    - **best_th** (*float*): Threshold for OOD detection.
    - **prec** (*float*): Precision at the threshold.
    - **recall** (*float*): Recall at the threshold.

### ncdia.algorithms.supervised.standard.py

Modules related to supervised learning

#### StandardSL

Class inherits from **BaseAlg**. Standard supervised learning algorithm

- <span class="highlight-text">**train_step(self, trainer, data, label, \*args, \*\*kwargs)**</span>

    Training step for standard supervised learning.

    **Parameters:**

    - **trainer** (*object*): Trainer object.
    - **data** (*torch.Tensor*): Input data.
    - **label** (*torch.Tensor*): Label data.
    - **args** (*tuple*): Additional arguments.
    - **kwargs** (*dict*): Additional keyword arguments.

    **Returns:**
    
    - **results** (*dict*): Training results. Contains the following keys:

        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.

- <span class="highlight-text">**val_step(self, trainer, data, label, \*args, \*\*kwargs)**</span>

    Validation step for standard supervised learning.

    **Parameters:**

    - **trainer** (*object*): Trainer object.
    - **data** (*torch.Tensor*): Input data.
    - **label** (*torch.Tensor*): Label data.
    - **args** (*tuple*): Additional arguments.
    - **kwargs** (*dict*): Additional keyword arguments.

    **Returns:**

    - **results** (*dict*): Validation results. Contains the following:

        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.

- <span class="highlight-text">**test_step(self, trainer, data, label, \*args, \*\*kwargs)**</span>

    Test step for standard supervised learning.

    **Parameters:**

    - trainer (object): Trainer object.
    - data (torch.Tensor): Input data.
    - label (torch.Tensor): Label data.
    - args (tuple): Additional arguments.
    - kwargs (dict): Additional keyword arguments.

    **Returns:**

    - **results** (*dict*): Test results. Contains the following:

        - **"loss"**: Loss value.
        - **"acc"**: Accuracy value.

## ncdia.dataloader

### ncdia.dataloader.base.py

- <span class="highlight-text">**build_dataloader(kwargs)**</span>

    Build data loader.

    **Parameters:**

    - **kwargs** (*dict*): Arguments for DataLoader. Contains the following:
        - **dataset** (*dict*): Dataset configuration.
        - other arguments for DataLoader, such as **batch_size**, **shuffle**, etc.

    **Returns:**

    - **loader** (*DataLoader*): Data loader.


### ncdia.dataloader.tools.py

Implements some of the commonly used dataloaders

### ncdia.dataloader.datasets

Implements some of the commonly used datasets, including:

- **CIFAR100**
- **CUB200**
- **Caltech101**
- **Food101**
- **ImageNet**
- **ImageNetR**
- **BM200**

### ncdia.dataloader.augmentations

Implements some of the commonly used augmentation methods.

## ncdia.model
### ncdia.models.models.py
- <span class="highlight-text">**get_network(config)**</span>


    load model.

    **Parameters:**
    -**trainer**(*config*): model config

### ncdia.models.net.inc_net.py
#### BaseNet

BaseNet for incremental learning.

- <span class="highlight-text">**\_\_init\_\_(self, network, base_classes, num_classes, att_classes, net_alice, mode)**</span>

    The constructor method that initializes an instance of **BaseNet**.

    **Parameters:**

    - **network** (*config*): The config of the network.
    - **base_classes**(*int*): The number of base classes.
    - **num_classes**(*int*): The total class number.
    - **att_classes**(*int*): The attribute class number.
    - **mode**(*str*): classifier mode.



- <span class="highlight-text">**feature_dim(self)**</span>

    The feature dimension of the network.

    **Returns:**

    - **out_dim**(*int*) feature dimension of the network.

- <span class="highlight-text">**extractor_vector(self, x)**</span>
    
    get features of input x.

    **Parameters:**

    - **x**(*tensor*): input data.

    **Returns:**

    - **out_features**(*tensor*) features of the input.

- <span class="highlight-text">**forward(self, x)**</span>
    
    forworad pass of the network.

    **Parameters:**

    - **x**(*tensor*): input data.


    **Returns:**
    
    - **results** (*dict*): forward pass results. Contains the following keys:
        - **"fmaps"**: [x_1, x_2, ..., x_n],
        - **"features"**: features
        - **"logits"**: logits


- <span class="highlight-text">**copy(self)**</span>

    copy.

    **Returns:**

    - **copy function**.



- <span class="highlight-text">**freeze(self)**</span>

    freeze parameters.

#### IncrementalNet

Incremental Network which follows BaseNet.


- <span class="highlight-text">**\_\_init\_\_(self, network, base_classes, num_classes, att_classes, net_alice, mode)**</span>

    The constructor method that initializes an instance of **BaseNet**.

    **Parameters:**

    - **network** (*config*): The config of the network.
    - **base_classes**(*int*): The number of base classes.
    - **num_classes**(*int*): The total class number.
    - **att_classes**(*int*): The attribute class number.
    - **mode**(*str*): classifier mode.

- <span class="highlight-text">**update_fc(self, nb_classes)**</span>

    update fc parameter, generate new fc and copy old parameter.

    **Parameters:**

    - **network** (*int*): New class number.

    **Returns:**

    - **fc**: updated fc layers.

- <span class="highlight-text">**generate_fc(self, in_dim, out_dim)**</span>

    **Parameters:**

    - **in_dim** (*int*): new fc in dimension.
    - **out_dim** (*int*): new fc out dimension.

    **Returns:**

    - **fc**: new fc layers.


- <span class="highlight-text">**forward(self, x)**</span>
    
    forworad pass of the network.

    **Parameters:**

    - **x**(*tensor*): input data.


    **Returns:**
    
    - **results** (*dict*): forward pass results. Contains the following keys:
        - **"fmaps"**: [x_1, x_2, ..., x_n],
        - **"features"**: features
        - **"logits"**: logits

- <span class="highlight-text">**weight_align(self, increment)**</span>
    
    normalize classifer parameters.

    **Parameters:**

    

    - **increment**(*int*): incremental classes.


    


## ncdia.trainers

### ncdia.trainers.base.py

#### BaseTrainer

Basic trainer class for training models.

**Attributes:**

- **model** (*nn.Module*): Neural network models.
- **train_loader** (*DataLoader*): DataLoader for training.
- **val_loader** (*DataLoader*): DataLoader for validation.
- **test_loader** (*DataLoader*): DataLoader for testing.
- **optimizer** (*Optimizer*): Optimizer.
- **scheduler** (*lr_scheduler._LRScheduler*): Learning rate scheduler.
- **criterion** (*Callable*): Criterion for training.
- **algorithm** (*object*): Algorithm for training.
- **metrics** (*dict*): Metrics for evaluation and testing.
- **session** (*int*): Session number.
- **max_epochs** (*int*): Total epochs for training.
- **max_train_iters** (*int*): Iterations on one epoch for training.
- **max_val_iters** (*int*): Iterations on one epoch for validation.
- **max_test_iters** (*int*): Iterations on one epoch for testing.
- **epoch** (*int*): Current training epoch.
- **iter** (*int*): Current iteration or index of the current batch.
- **cfg** (*Configs*): Configuration for trainer.
- **hooks** (*List[Hook]*): List of registered hooks.
- **logger** (*Logger*): Logger for logging information.
- **device** (*torch.device*): Device to use.
- **work_dir** (*str*): Working directory to save logs and checkpoints.
- **exp_name** (*str*): Experiment name.
- **load_from** (*str*): Checkpoint file path to load.

**Methods:**

- <span class="highlight-text">** \_\_init\_\_(self, cfg, session, model, train_loader, val_loader, test_loader, default_hooks, custom_hooks, load_from, exp_name, work_dir)**</span>

    The constructor method that initializes an instance of **BaseTrainer**. 

    **Parameters:**

    - **cfg** (*dict, optional*): Configuration for trainer, Contains:
        - **'trainer'** (*dict*):
            - 'type' (*str*): Type of trainer.
        - **'algorithm'** (*dict*):
            - 'type' (*str*): Type of algorithm.
        - **'criterion'** (*dict*):
            - 'type' (*str*): Type of criterion for training.
        - **'optimizer'**:
            - 'type' (*str*): Name of optimizer.
            - 'param_groups' (*dict | None*): If provided, directly optimize
                param_groups and abandon model.
            - kwargs (*dict*) for optimizer, such as 'lr', 'weight_decay', etc.
        - **'scheduler'**:
            - 'type' (*str*): Name of scheduler.
            - kwargs (*dict*) for scheduler, such as 'step_size', 'gamma', etc.
        - **'device' **(*str | torch.device | None*): Device to use.
            If None, use 'cuda' if available.
        - **'trainloader'**:
            - 'dataset': 
                - 'type' (*str*): Type of dataset.
                - kwargs (*dict*) for dataset, such as 'root', 'split', etc.
            - kwargs (*dict*) for DataLoader, such as 'batch_size', 'shuffle', etc.
        - **'valloader'**:
            - 'dataset': 
                - 'type' (*str*): Type of dataset.
                - kwargs (*dict*) for dataset, such as 'root', 'split', etc.
            - kwargs (*dict*) for DataLoader, such as 'batch_size', 'shuffle', etc.
        - **'testloader'**:
            - 'dataset':
                - 'type' (*str*): Type of dataset.
                - kwargs (*dict*) for dataset, such as 'root', 'split', etc.
            - kwargs (*dict*) for DataLoader, such as 'batch_size', 'shuffle', etc.
        - **'exp_name'** (*str*): Experiment name.
        - **'work_dir'** (*str*): Working directory to save logs and checkpoints.
    - **session** (*int*): Session number. If == 0, execute pre-training.
        If > 0, execute incremental training.
    - **model** (*nn.Module*): Model to be trained.
    - **train_loader** (*DataLoader | dict, optional*): DataLoader for training.
    - **val_loader** (*DataLoader | dict, optional*): DataLoader for validation.
    - **test_loader** (*DataLoader | dict, optional*): DataLoader for testing.
    - **default_hooks** (*dict, optional*): Default hooks to be registered.
    - **custom_hooks** (*list, optional*): Custom hooks to be registered.
    - **load_from** (*str, optional*): Checkpoint file path to load.
    - **work_dir** (*str, optional*): Working directory to save logs and checkpoints.

- <span class="highlight-text">** train_step(self, batch, \*\*kwargs)**</span>

    Training step. **This method should be implemented in subclasses.**

    **Parameters:**

    - **batch** (*dict | tuple | list*): A batch of data from the data loader.

    **Returns:**

    - results (dict): Contains the following:

        {"key1": value1, "key2": value2,...}

        keys denote the description of the value, such as **"loss"**, **"acc"**, **"ccr"**, etc.
        values are the corresponding values of the keys, can be *int*, *float*, *str*, etc.    

- <span class="highlight-text">** val_step(self, batch, **kwargs)**</span>

    Validation step. **This method should be implemented in subclasses.**

    **Parameters:**

    - **batch** (*dict | tuple | list*): A batch of data from the data loader.

    **Returns:**

    - results (dict): Contains the following:

        {"key1": value1, "key2": value2,...}

        keys denote the description of the value, such as **"loss"**, **"acc"**, **"ccr"**, etc.
        values are the corresponding values of the keys, can be *int*, *float*, *str*, etc.    

- <span class="highlight-text">** test_step(self, batch, **kwargs)**</span>

    Test step. **This method should be implemented in subclasses.**

    **Parameters:**

    - **batch** (*dict | tuple | list*): A batch of data from the data loader.

    **Returns:**

    - results (dict): Contains the following:

        {"key1": value1, "key2": value2,...}

        keys denote the description of the value, such as **"loss"**, **"acc"**, **"ccr"**, etc.
        values are the corresponding values of the keys, can be *int*, *float*, *str*, etc.   

- <span class="highlight-text">** train(self)**</span>

    Launch the training process.

    **Returns:**

    -  **model** (*nn.Module*): Trained model.

- <span class="highlight-text">** val(self)**</span>

Validation process.

- <span class="highlight-text">** test(self)**</span>

Test process.

- <span class="highlight-text">** load_ckpt(self, fpath, device='cpu')**</span>

    Load checkpoint from file.

    **Parameters:**

    - **fpath** (*str*): Checkpoint file path.
    - **device** (*str*): Device to load checkpoint. Defaults to 'cpu'.

    **Returns:**

    - **model** (*nn.Module*): Loaded model.

- <span class="highlight-text">** save_ckpt(self, fpath)**</span>

    Save checkpoint to file.

    **Parameters:**

    - **fpath** (*str*): Checkpoint file path.

- <span class="highlight-text">** call_hook(self, fn_name: str, **kwargs)**</span>

    Call all hooks with the specified function name.

    **Parameters:**

    - **fn_name** (*str*): Function name to be called, such as:

        - **'before_train_epoch'**
        - **'after_train_epoch'**
        - **'before_train_iter'**
        - **'after_train_iter'**
        - **'before_val_epoch'**
        - **'after_val_epoch'**
        - **'before_val_iter'**
        - **'after_val_iter'**
    
    - **kwargs** (*dict*): Arguments for the function.

- <span class="highlight-text">**register_hook(self, hook, priority=None)**</span>

    **Register a hook into the hook list.**

    The hook will be inserted into a priority queue, with the specified priority (See :class:`Priority` for details of priorities). For hooks with the same priority, they will be triggered in the same order as they are registered. Priority of hook will be decided with the following priority:

    - ``priority`` argument. If ``priority`` is given, it will be priority of hook.
    - If ``hook`` argument is a dict and ``priority`` in it, the priority will be the value of ``hook['priority']``.
    - If ``hook`` argument is a dict but ``priority`` not in it or ``hook`` is an instance of ``hook``, the priority will be ``hook.priority``.

    **Parameters:**

    - **hook** (*:obj:`Hook` or dict*): The hook to be registered. priority (int or str or :obj:`Priority`, optional): Hook priority. Lower value means higher priority.

- <span class="highlight-text">** register_default_hooks(self, hooks=None)**</span>

    **Register default hooks into hook list.**
    
    ``hooks`` will be registered into runner to execute some default actions like updating model parameters or saving checkpoints.

    Default hooks and their priorities:

    | Hooks               | Priority           |
    | :-------------------| :------------------|
    | RuntimeInfoHook     |  VERY_HIGH (10)    |
    | IterTimerHook       |  NORMAL (50)       |
    | DistSamplerSeedHook |  NORMAL (50)       |
    | LoggerHook          |  BELOW_NORMAL (60) |
    | ParamSchedulerHook  |  LOW (70)          |
    | CheckpointHook      |  VERY_LOW (90)     |

    If ``hooks`` is None, above hooks will be registered by default:

        default_hooks = dict(
            logger=dict(type='LoggerHook'),
            model=dict(type='ModelHook'),
            alg=dict(type='AlgHook'),
            optimizer = dict(type='OptimizerHook'),
            scheduler = dict(type='SchedulerHook'),
            metric = dict(type='MetricHook'),
        )

    If not None, ``hooks`` will be merged into ``default_hooks``.
    If there are None value in default_hooks, the corresponding item will
    be popped from ``default_hooks``:

        hooks = dict(timer=None)

    The final registered default hooks will be :obj:`RuntimeInfoHook`, :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`, :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

    **Parameters:**

    - **hooks** (*dict[str, Hook or dict]*): Default hooks or configs to be registered.

- <span class="highlight-text">** register_custom_hooks(self, hooks)**</span>

    Register custom hooks into hook list.

    **Parameters:**

    **hooks** (*list[Hook | dict]*): List of hooks or configs to be registered.

- <span class="highlight-text">** register_hooks(self, default_hooks=None, custom_hooks=None)**</span>

    Register default hooks and custom hooks into hook list.

    **Parameters:**

    - **default_hooks** (*dict[str, dict] or dict[str, Hook]*): Hooks to execute default actions like updating model parameters and saving checkpoints.  Defaults to None.
    - **custom_hooks** (*list[dict] or list[Hook]*): Hooks to execute custom actions like visualizing images processed by pipeline. Defaults to None.

- <span class="highlight-text">**get_hooks_info(self)**</span>

    Get registered hooks information.

    **Returns:**

    - **info** (*str*): Information of registered hooks.

### ncdia.trainers.pretrainer.py

#### PreTrainer

PreTrainer class for pre-training a model on session 0.

**Attributes:**

- **max_epochs** (*int*): Total epochs for training.

**Methods:**

- **\_\_init\_\_(self, max_epochs=1, \*\*kwargs):** The constructor method that initializes an instance of **PreTrainer**. **max_epochs** (*int*): Total epochs for training.
- **train_step(self, batch, \*\*kwargs):** Training step.
- **val_step(self, batch, \*\*kwargs):** Validation step.
- **test_step(self, batch, \*\*kwargs):** Test step.
- **batch_parser(batch)** 
    
    Parse a batch of data.

    **Parameters:**

    - **batch** (*dict | tuple | list*): A batch of data.

    **Returns:**

    - **data** (*torch.Tensor | list*): Input data.
    - **label** (*torch.Tensor | list*): Label data.
    - **attribute** (*torch.Tensor | list*): Attribute data.
    - **imgpath** (*list of str*): Image path.

### ncdia.trainers.inctrainer.py

#### IncTrainer

IncTrainer class for incremental training.

**Attributes:**

- **sess_cfg** (*Configs*): Session configuration.
- **num_sess** (*int*): Number of sessions.
- **session** (*int*): Session number. If == 0, execute pre-training.
    If > 0, execute incremental training.
- **hist_trainset** (*MergedDataset*): Historical training dataset.
- **hist_valset** (*MergedDataset*): Historical validation dataset.
- **hist_testset** (*MergedDataset*): Historical testing dataset.

**Methods:**

- <span class="highlight-text">**  \_\_init\_\_(self, cfg=None, sess_cfg=None, ncd_cfg=None, session=0, model=None, hist_trainset=None, hist_testset=None, old_model=None, \*\*kwargs)**</span>

    The constructor method that initializes an instance of **IncTrainer**. 

    **Parameters:**

    - **model** (*nn.Module*): Model to be trained.
    - **cfg** (*dict*): Configuration for trainer.
    - **sess_cfg** (*Configs*): Session configuration.
    - **session** (*int*): Session number. Default: 0.

- <span class="highlight-text">** train(self)**</span>

    Incremental training. 

    **self.num_sess** determines the number of sessions, and session number is stored in **self.session**.

    **Returns:**

    - **model** (*nn.Module*): Trained model.

### ncdia.trainers.hooks

Implements some of the commonly used hooks.

#### Hook

*ncdia.trainers.hooks.hook.py*

Base hook class. All hooks should inherit from this class.

#### AlgHook

*ncdia.trainers.hooks.alghook.py*

A hook to modify algorithm state in the pipeline. This class is a base class for all algorithm hooks.

#### LoggerHook

*ncdia.trainers.hooks.loggerhook.py*

A hook to log information during training and evaluation.

#### MetricHook

*ncdia.trainers.hooks.metrichook.py*

A hook to calculate metrics during evaluation and testing.

#### ModelHook

*ncdia.trainers.hooks.modelhook.py*

A hook to change model state in the pipeline, such as setting device, changing model to eval mode, etc.

#### NCDHook

*ncdia.trainers.hooks.ncdhook.py*

A hook to execute OOD and NCD detection to relabel data

#### OptimizerHook

*ncdia.trainers.hooks.optimizerhook.py*

A hook to put optimizer to zero_grad and step during training.

#### SchedulerHook

*ncdia.trainers.hooks.schedulerhook.py*

A hook to change learning rate during training.

### ncdia.trainers.optims

#### ncdia.trainers.optims.optimizer.py

- <span class="highlight-text">**build_optimizer(type, model, param_groups=None, \*\*kwargs)**</span>

    Build optimizer.

    **Parameters:**

    - **type** (*str*): type of optimizer
    - **model** (*nn.Module | dict*): model or param_groups
    - **param_groups** (*dict | None*): 
        if provided, directly optimize param_groups and abandon model
    - **kwargs** (*dict*): arguments for optimizer

    **Returns:**

    - **optimizer** (*torch.optim.Optimizer*): optimizer

#### ncdia.trainers.optims.scheduler.py

Implements some of the commonly used scheduler.

- **CosineWarmupLR**
- **LinearWarmupLR**
- **ConstantLR**

**Methods:**

- <span class="highlight-text">**build_scheduler(type, optimizer, \*\*kwargs)**</span>

    Build learning rate scheduler.

    **Parameters:**

    - **type** (*str*): type of scheduler
    - **optimizer** (t*orch.optim.Optimizer*): optimizer
    - **kwargs** (*dict*): arguments for scheduler

    **Returns:**

    - **lr_scheduler** (*torch.optim.lr_scheduler._LRScheduler*): learning rate scheduler

### ncdia.trainers.priority

Hook priority levels.

#### Priority

| Level        | Value  |
| :------------| :------|
| HIGHEST      | 0      |
| VERY_HIGH    | 10     |
| HIGH         | 30     |
| ABOVE_NORMAL | 40     |
| NORMAL       | 50     |
| BELOW_NORMAL | 60     |
| LOW          | 70     |
| VERY_LOW     | 90     |
| LOWEST       | 100    |

## ncdia.utils
