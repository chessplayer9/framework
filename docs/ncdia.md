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


- <span class="highlight-text">** cfg(self)**</span>

Configuration for trainer

- <span class="highlight-text">** hooks(self)**</span>

List of registered hooks

- <span class="highlight-text">** logger(self)**</span>

Logger for logging information

- <span class="highlight-text">** work_dir(self)**</span>

 Working directory to save logs and checkpoints

- <span class="highlight-text">** session(self)**</span>

Session number. If == 0, execute pre-training.

- <span class="highlight-text">** model(self)**</span>

- <span class="highlight-text">** train_loader(self)**</span>

- <span class="highlight-text">** val_loader(self)**</span>

- <span class="highlight-text">** test_loader(self)**</span>

- <span class="highlight-text">** optimizer(self)**</span>

- <span class="highlight-text">** scheduler(self)**</span>

- <span class="highlight-text">** criterion(self)**</span>

- <span class="highlight-text">** algorithm(self)**</span>

- <span class="highlight-text">** metrics(self)**</span>

- <span class="highlight-text">** max_epochs(self)**</span>

- <span class="highlight-text">** max_train_iters(self)**</span>

- <span class="highlight-text">** max_val_iters(self)**</span>

- <span class="highlight-text">** max_test_iters(self)**</span>

- <span class="highlight-text">** device(self)**</span>

- <span class="highlight-text">** train_step(self, batch, **kwargs)**</span>

- <span class="highlight-text">** val_step(self, batch, **kwargs)**</span>

- <span class="highlight-text">** test_step(self, batch, **kwargs)**</span>

- <span class="highlight-text">** train(self)**</span>

- <span class="highlight-text">** val(self)**</span>

- <span class="highlight-text">** test(self)**</span>

- <span class="highlight-text">** load_ckpt(self, fpath, device='cpu')**</span>

- <span class="highlight-text">** save_ckpt(self, fpath)**</span>

- <span class="highlight-text">** call_hook(self, fn_name: str, **kwargs)**</span>

- <span class="highlight-text">**register_hook(self, hook, priority=None)**</span>

- <span class="highlight-text">** register_default_hooks(self, hooks=None)**</span>

- <span class="highlight-text">** register_custom_hooks(self, hooks)**</span>

- <span class="highlight-text">** register_hooks(self, default_hooks=None, custom_hooks=None)**</span>

- <span class="highlight-text">**get_hooks_info(self)**</span>

## ncdia.utils
