# CausalBench ICLR-23 Challenge Starter Repository

![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.0.0-blue)

The starter repository for submissions to the [CausalBench challenge](https://www.gsk.ai/causalbench-challenge/) for gene-gene graph inference from genetic perturbation experiments.

[CausalBench](https://arxiv.org/abs/2210.17283) is a comprehensive benchmark suite for evaluating network inference methods on perturbational single-cell gene expression data. 
CausalBench introduces several biologically meaningful performance metrics and operates on two large, curated and openly available benchmark data sets for evaluating methods on the inference of gene regulatory networks from single-cell data generated under perturbations.

## Install

```bash
pip install -r requirements.txt
```

## Use

### Setup

- Create a cache directory. This will hold any preprocessed and downloaded datasets for faster future invocation.
  - `$ mkdir /path/to/causalbench_cache`
  - _Replace the above with your desired cache directory location._
- Create an output directory. This will hold all program outputs and results.
  - `$ mkdir /path/to/causalbench_output`
  - _Replace the above with your desired output directory location._
- Create a plot directory. This will hold all plots and final metrics of your experiments.
  - `$ mkdir /path/to/plots`
  - _Replace the above with your desired plots directory location._


### How to Run the Full Benchmark Suite?

Example of command to run a model on the rpe1 dataset in the interventional regime. We recommand running this simple model first such that all the dataset files get downloaded and cached.

```bash
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory /path/to/output/ \
    --data_directory /path/to/data/storage \
    --training_regime partial_interventional \
    --partial_intervention_seed 0 \
    --fraction_partial_intervention 0.5 \
    --model_name random100 \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter
```

Results are written to the folder at `/path/to/output/`, and processed datasets will be cached at `/path/to/data/storage`. The above command run the model on a dataset containing both observational data (i.e., samples where no gene was perturbed), and some interventional data (one gene was perturbed in each sample). Here, given `--fraction_partial_intervention 0.5`, 50% of the genes are intervenened on in some samples. This subset of genes is randomly selected, using the `--partial_intervention_seed`. See below for the exact parameters your submission will be evaluated on.

### How to Evaluate a Custom Graph Inference Function?

To run a custom graph inference function, set `--model_name="custom"` and `--inference_function_file_path` to the file path that contains your custom graph inference function (e.g. [grnboost.py](../causalbench-starter-main%205/src/grnboost.py) in this repo). You are given two starter implementations to choose from in src/, grnboost.py and dcdi.py. Your mission is to choose one of them and fine tune them to improve their performance. Hints on potential ways to improve the methods can be found directly in the code. 

You should evaluate your method with the following command:

```bash
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory /path/to/output/ \
    --data_directory /path/to/data/storage \
    --training_regime partial_interventional \
    --partial_intervention_seed 0 \
    --fraction_partial_intervention $FRACTION \
    --model_name custom \
    --inference_function_file_path /path/to/custom_inference_function.py \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter
```

...where `"path/to/custom_inference_function.py"` contains code for your custom graph inference function corresponding to the [AbstractInferenceModel interface](https://github.com/causalbench/causalbench/blob/master/causalscbench/models/abstract_model.py). You should run your method four times, by sweeping the `$FRACTION` parameter in {0.25, 0.5, 0.75, 1.0}. For experimentation with limited resources or to reduce runtime, you may want to set the `--subset_data` parameter to a smaller value. To run the dcdi.py model, a GPU is almost a requirement. On the other hand, grnboost can be run on CPU alone.

For this challenge, you are restricted to use one of the two functions that are given to you. Almost any modifications are allowed, even though the final algorithm should remain close in essence to the baseline method. You can also directly modify the algorithm. For example, you could write new probability distribution estimators for DCDI like [here](https://github.com/causalbench/causalbench/tree/master/causalscbench/third_party/dcdi/dcdi/models). All your code needs to be in /src, either directly in e.g. `dcdi.py` or as subpackages, and should run properly using the above command. See the section below expliciting more rules and information on what modification are allowed or not. 

### How to plot and score your custom method?

For your convinience, we include a script in /scripts, plots.py, for you to plot your custom function performance, as well as compute your final area under curve score. To run it, simply use the following command `$ python scripty/plots.py /path/to/plots /path/to/causalbench_output`. Two files will be saved in `/path/to/plots`, `sweeps_partial_interventions.pdf`and `auc_scores.json`. The pdf contains a figure plot and the json contains models scores.

## Submission instructions

For submission, you will need the eval.ai-CLI tool, which can be installed with pip
  - [EvalAI-CLI Documentation](https://cli.eval.ai/)
  - Run `$ pip install evalai`

You will also need Docker, as submissions must be submitted as container images. We have provided an adequate default Dockerfile for your convenience.
  - [Docker Installation Instructions](https://docs.docker.com/desktop/)

Please note that all your submitted code must either be loaded via a dependency in `requirements.txt` or be present in the `src/` directory in this starter repository for the submission to succeed.


### Submission steps

- Navigate to the directory to which you have cloned this repo to.
  - `$ cd /path/to/causalbench-starter`
- Ensure you have ONE graph inference function (inheriting from `AbstractInferenceModel`) in [main.py](src/main.py). For example copy your final method to main.py `$ cp src/grnboost.py src/main.py`.
  - _This is your pre-defined program entry point._
- Build your container image with the provided Dockerfile
  - `$ docker build -t submission:latest .`
  - `$ IMAGE = "submission:latest"` 
- Use the EvalAI-CLI command to submit your image
  - Run the following command to submit your container image:
    - `$ evalai push $IMAGE --phase gsk-causalbench-test-1981`
    - **Please note** that you have a maximum number of submissions that any submission will be counted against.
    - **Please note** that you will not receive your final private leaderboard score until the challenge period is over.

That’s it! Our pipeline will take your image and test your function. If your method violates one of the requirements (graph of at least 1,000 edges, no duplicate edges), you will be informed of this failure.

## Rules and information on submission evaluation

- The challenge private leaderboard will be evaluated on a different dataset than the one available to participants locally.
- Exploiting knowledge of the evaluation dataset or of the evaluation metric in your method is forbidden.
- Using external data sources of known gene-gene intercations is forbidden.
- Only methods that improve on the given baseline code will be considered.
- The size of the output graph should always be at least 1,000 edges.
- Your method should show a clear improvement given more perturbation data (increasing curve) and the final private score will take this into account, in addition to the AUC score.


If you have any questions or concerns, please reach out to us at [rd.causalbench-challenge@gsk.com](mailto:rd.causalbench-challenge@gsk.com?subject=[CausalBench-Support-Request])


### Frequently Asked Questions (FAQ)

#### _"Will there be a public challenge leaderboard?"_

No. Participants are asked to compare their solutions internally against the provided baselines in the `causalbench` repository.
A final private leaderboard will be created using the submissions received via `eval.ai` after the challenge submission period is closed.

#### _"Which specific parts of the whole app lifecycle are we allowed to change?"_

You are able to change the graph inference function only (please see instructions above). We have chosen to fix the hyperparameters in order to enable comparability of the developed solutions.

#### _"How will submissions be scored?"_

Your submission will be evaluated by running it with different level of interventional data (25%, 50%, 75%, 100%, see parameter '--fraction_partial_intervention') and the mean wasserstein distance score outputted in the metrics file will be used. A good model, when the mean wassertein distance is plotted against the level of interventional data, should exhibit an upward trend. 

#### _"How do you define the best submission? Is it the last one or the best from all submitted?"_

We will use the last submission to calculate the team's score to avoid conferring an advantage to teams that produce more submissions.

#### _"The submissions limit (3) seems to be quite low. "_

The idea is for participants to develop and evaluate their own solutions internally against the many existing baselines already implemented in [CausalBench](https://github.com/causalbench/causalbench) - hence there is no public leaderboard.
There will be a final private leaderboard that we will score through the eval.ai submission system.


## Citation

Please consider citing, if you reference or use our methodology, code or results in your work:

    @article{chevalley2022causalbench,
        title={{CausalBench: A Large-scale Benchmark for Network Inference from Single-cell Perturbation Data}},
        author={Chevalley, Mathieu and Roohani, Yusuf and Mehrjou, Arash and Leskovec, Jure and Schwab, Patrick},
        journal={arXiv preprint arXiv:2210.17283},
        year={2022}
    }


### License

[License](LICENSE.txt)

### Authors

Mathieu Chevalley, GSK plc and ETH Zurich<br/>
Jacob A. Sackett-Sanders, GSK plc<br/>
Yusuf H Roohani, GSK plc and Stanford University<br/>
Arash Mehrjou, GSK plc<br/>
Patrick Schwab, GSK plc<br/>

### Acknowledgements

MC, YR, JSS, AM and PS are employees and shareholders of GlaxoSmithKline plc.