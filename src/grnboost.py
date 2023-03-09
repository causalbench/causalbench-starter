"""
Copyright 2023  Mathieu Chevalley, Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple

import distributed
import numpy as np
from arboreto import algo
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes


class GRNBoost(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.n_workers = 20
        self.threads_per_worker = 10
        self.gene_expression_threshold = 0.25

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        """
            expression_matrix: numpy array of size n_samples x n_genes, which contains the expression values
                                of each gene in different cells
            interventions: list of size n_samples. Indicates which gene has been perturbed in each sample.
                            If value is "non-targeting", no gene was targeted (observational sample).
                            If value is "excluded", a gene was perturbed which is not in gene_names (a confounder was perturbed).
                            You may want to exclude those samples or still try to leverage them.
            gene_names: names of the genes of size n_genes. To be used as node names for the output graph.


        Returns:
            List of string tuples: output graph as list of edges. 
        """
        # We remove genes that have a non-zero expression in less than 25% of samples.
        # You may want to select the genes differently.
        # You could also preprocess the expression matrix, for example to impute 0.0 expression values.
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix,
            gene_names,
            expression_threshold=self.gene_expression_threshold,
        )
        local_cluster = distributed.LocalCluster(
            n_workers=self.n_workers, threads_per_worker=self.threads_per_worker
        )
        custom_client = distributed.Client(local_cluster)

        # The GRNBoost algo was tailored for only observational data.
        # You may want to modify the algo to take into account the perturbation information in the "interventions" input.
        # This may be achieved by directly modifying the algorithm or by modulating the expression matrix that is given as input.
        network = algo.grnboost2(
            expression_data=expression_matrix,
            gene_names=gene_names,
            client_or_address=custom_client,
            seed=seed,
            early_stop_window_length=15,
            verbose=True,
        )

        # You may want to postprocess the output network to select the edges with stronger expected causal effects.
        return [(i, j) for i, j in network[["TF", "target"]].values]
