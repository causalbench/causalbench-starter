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
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import List, Literal, Tuple

import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network,
    remove_lowly_expressed_genes,
)
from causalscbench.third_party.dcdi.dcdi.data import DataManagerFile
from causalscbench.third_party.dcdi.dcdi.models.flows import DeepSigmoidalFlowModel
from causalscbench.third_party.dcdi.dcdi.models.learnables import (
    LearnableModel_NonLinGaussANM,
)
from causalscbench.third_party.dcdi.dcdi.train import train


class DCDI(AbstractInferenceModel):
    # You can choose between "DCDI-G" (non linear gaussian model) and "DCDI-DSF" (deep sigmoidal flow)
    MODEL_NAME: Literal["DCDI-DSF", "DCDI-G"] = "DCDI-DSF"

    def __init__(self) -> None:
        super().__init__()

        # Default from DCDI, can be modified and optimized. For more intuition about the parameters, please read the DCDI paper (https://arxiv.org/abs/2007.01754)
        self.opt = SimpleNamespace()
        self.opt.train_patience = 5
        self.opt.train_patience_post = 5
        self.opt.num_train_iter = 30000
        self.opt.no_w_adjs_log = True
        self.opt.mu_init = 1e-8
        self.opt.gamma_init = 0.0
        self.opt.optimizer = "rmsprop"
        self.opt.lr = 1e-2
        self.opt.train_batch_size = 64
        self.opt.reg_coeff = 0.1
        self.opt.coeff_interv_sparsity = 0
        self.opt.stop_crit_win = 100
        self.opt.h_threshold = 1e-8
        self.opt.omega_gamma = 1e-4
        self.opt.omega_mu = 0.9
        self.opt.mu_mult_factor = 2
        self.opt.lr_reinit = 1e-2
        self.opt.intervention = True
        self.opt.intervention_type = "perfect"
        self.opt.intervention_knowledge = "known"
        self.opt.gpu = True

        self.gene_expression_threshold = 0.25
        self.soft_adjacency_matrix_threshold = 0.5

        self.fraction_train_data = 0.8

        self.gene_partition_sizes = 50
        self.max_parallel_executors = 8

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
        gene_names = np.array(gene_names)

        def process_partition(partition):
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            node_dict = {g: idx for idx, g in enumerate(gene_names_)}
            gene_names_set = set(gene_names_)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            expression_matrix_ = expression_matrix_[subset, :]
            gene_to_interventions = dict()
            for i, intervention in enumerate(interventions_):
                gene_to_interventions.setdefault(intervention, []).append(i)

            mask_intervention = []
            regimes = []
            regime_index = 0
            start = 0
            data = np.zeros_like(expression_matrix_)
            for inv, indices in gene_to_interventions.items():
                targets = [] if inv == "non-targeting" else [node_dict[inv]]
                regime = 0 if inv == "non-targeting" else regime_index + 1
                mask_intervention.extend([targets for _ in range(len(indices))])
                regimes.extend([regime for _ in range(len(indices))])
                end = start + len(indices)
                data[start:end, :] = expression_matrix_[indices, :]
                start = end
                if inv != "non-targeting":
                    regime_index += 1

            regimes = np.array(regimes)

            train_data = DataManagerFile(
                data,
                mask_intervention,
                regimes,
                self.fraction_train_data,
                train=True,
                normalize=False,
                random_seed=seed,
                intervention=True,
                intervention_knowledge="known",
            )
            test_data = DataManagerFile(
                data,
                mask_intervention,
                regimes,
                self.fraction_train_data,
                train=False,
                normalize=False,
                random_seed=seed,
                intervention=True,
                intervention_knowledge="known",
            )

            # You may want to play around with the hyper parameters to find the optimal ones.
            if DCDI.MODEL_NAME == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(
                    num_vars=len(gene_names_),
                    num_layers=2,
                    hid_dim=15,
                    intervention=True,
                    intervention_type=self.opt.intervention_type,
                    intervention_knowledge=self.opt.intervention_knowledge,
                    num_regimes=train_data.num_regimes,
                )
            elif DCDI.MODEL_NAME == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(
                    num_vars=len(gene_names_),
                    cond_n_layers=2,
                    cond_hid_dim=15,
                    cond_nonlin="leaky-relu",
                    flow_n_layers=2,
                    flow_hid_dim=10,
                    intervention=True,
                    intervention_type=self.opt.intervention_type,
                    intervention_knowledge=self.opt.intervention_knowledge,
                    num_regimes=train_data.num_regimes,
                )
            else:
                raise ValueError("Model has to be in {DCDI-G, DCDI-DSF}")

            train(model, train_data, test_data, self.opt)

            adjacency = model.get_w_adj()
            # The soft adjacency matrix is currently thresholded at 0.5 to consider an edge as true positive.
            # You can change the threshold or find smarter ways to select edges out of the soft adjacency matrix.
            indices = np.nonzero(adjacency > self.soft_adjacency_matrix_threshold)
            edges = set()
            for (i, j) in indices:
                edges.add((gene_names_[i], gene_names_[j]))
            return list(edges)

        # DCDI can't handle the full graph as it does not scale well in terms of number of nodes.
        # You can work on improving its scaling properties such that is handles larger graphs.
        # Currently: genes are partitioned into random independent sub-graphs.
        # You may find smarter ways to partition/group or select the genes.
        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_executors) as executor:
            partition_results = list(executor.map(process_partition, partitions))
            for result in partition_results:
                edges += result
        return edges
