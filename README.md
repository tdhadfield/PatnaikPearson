# Patnaik-Pearson intrinsic dimension for internal representations of neural networks

This repo contains source code and notebooks for the numerical results presented in:

https://arxiv.org/abs/2606.19268

Patnaik-Pearson intrinsic dimension for internal representations of neural networks

We define a new measure of intrinsic dimension of a data manifold, which we call the Patnaik-Pearson dimension, and apply this to internal representations of neural networks, in particular transformers. The inspiration for this comes from the HTSR and SETOL work of Martin, Mahoney and Hinrichs, combined with the TwoNN intrinsic dimension estimator of Facco et al. We prove various properties of this intrinsic dimension estimator. Treating weight matrices of neural networks as data manifolds, for weight matrices whose Empirical Spectral Density follows a Pareto (Power Law) distribution, we relate the Patnaik-Pearson dimension to the HTSR and SETOL analysis, and show that critical values of the tail exponent coincide for the two approaches. Using a combination of theoretical and numerical techniques, we study the behaviour of the Patnaik-Pearson dimension of a data manifold under the transformations typical to neural networks. We apply this machinery to the BERT-base and DeepSeek-R1-Distill-Qwen-1 models, to investigate first the Patnaik-Pearson dimension of the initial data manifold of token embeddings, and second the evolution of the Patnaik-Pearson dimension as token embeddings pass through the layers of the model.