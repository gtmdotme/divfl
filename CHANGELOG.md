# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- porting from `tf.compact.v1` &rarr; `tf2`
- decoupling dataset, clients, model from FederatedBase
- TODO: docstrings in standard format
- TODO: comments
- dockerfile, conda, venv
- abstract classes
- power-of-choice not properly seeded

## [1.1.1] - 2023-08-18

### Changed

- Refactoring code:
    - self.latest_model &rarr; self.latest_model_params
    - train_metrics() &rarr; evaluate(mode='train')
    - test_metrics() &rarr; evaluate(mode='test')
    - self.updatevec commented since not used anywhere
- updated `Metrics` class to include more metrics
- added more evaluation metrics
- added comments

## [1.1.0] - 2023-08-12

### Changed

- Major refactoring of the code
    - `''' docstring '''` &rarr; `""" docstring """`
    - added logger (.info, .debug) at a few places
    - In `main.py`:
        - learner &rarr; Model
        - optimizer &rarr; Trainer
        - params &rarr; hyper_params
        - options &rarr; hyper_params
        - read_options() &rarr; read_inputs()
    - In Model classes:
        - solve_inner() &rarr; train_for_epochs()
        - solve_iters() &rarr; train_for_iters()
        - test() &rarr; evaluate()
        - soln &rarr; model_params
    - In `client.py`:
        - solve_inner() &rarr; train_for_epochs()
        - solve_iters() &rarr; train_for_iters()
        - train_error_and_loss() &rarr; train_metrics()
        - test() &rarr; test_metrics()
        - eval_data &rarr; test_data
        - num_samples &rarr; num_train_samples
        - test_samples &rarr; num_test_samples
        - soln &rarr; model_params
    - In Trainer classes:
        - train_error_and_loss() &rarr; train_metrics()
        - test() &rarr; test_metrics()
        - params &rarr; hyper_params
        - learner &rarr; Model
        - tqdm.write() &rarr; print()
    - In `*scripts.sh`:
        - `--optimizer` &rarr; `--trainer`
- Added report for tf1&rarr;tf2 migration
- Minor cosmetic changes

## [1.0.2] - 2023-06-28

### Changed

- Documented setting up the conda environment
- .gitignore updated and __pycache__ files removed
- Minor decorative changes
- Fixed issue of order of imports (put sklearn before tensorflow)

## [1.0.1] - 2023-05-31

### Changed

- Migrated from tf1 to tf2
- Resolved packages issues and added support for conda
- Uploaded preprocessed data to onedrive
- Minor issues resolved

## [1.0.0] - 2023-05-25

### Added

- Forked from [DivFL](https://github.com/melodi-lab/divfl)
