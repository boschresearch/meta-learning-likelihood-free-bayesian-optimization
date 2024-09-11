# MALIBO: Meta-learning for Likelihood-free Bayesian Optimization

This repository is an original implementation for our ICML 2024 paper: [MALIBO: Meta-learning for Likelihood-free Bayesian Optimization](https://proceedings.mlr.press/v235/pan24b.html) by Jiarong Pan, Stefan Falkner, Felix Berkenkamp and Joaquin Vanschoren. The code allows the users to use our implementation of MALIBO, and to reproduce the results for Forrester function and HPO-B benchmarks in the paper.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication. It will neither be maintained nor monitored in any way.

## Setup

```bash
conda create env -f environment.yml
conda activate malibo
```

## Run

To run the optimization for Forrester function:

```bash
python run_forrester.py
```

To run HPO-B benchmark:

```bash
cd benchmarks
git clone https://github.com/machinelearningnuremberg/HPO-B.git
```

You need to download the HPO-B benchmarks data and put it into `benchmarks/HPO-B/hpob-data`, after that you can run:

```bash
python run_hpob.py --search_space_id 4796 --test_seed test0 --no-continuous --evaluations 100 --output results/hpob
```

You can visualize the generated results using the plotting functions in HPO-B.

```bash
python benchmarks/generate_json_hpob.py
python benchmarks/benchmark_plot_hpob.py
```

## Cite

If you find this code useful in your research, please cite the paper:

```latex
@inproceedings{pan2024malibo,
  title = {{MALIBO}: Meta-learning for Likelihood-free {B}ayesian Optimization},
  author = {Pan, Jiarong and Falkner, Stefan and Berkenkamp, Felix and Vanschoren, Joaquin},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {39102--39134},
  year = {2024},
}
```

## License

`MALIBO` is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in `MALIBO`, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).
