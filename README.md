# Adaptive Political Surveys and GPT-4

This repository contains the code and data to reproduce the analysis in "Adaptive political surveys and GPT-4: Tackling the cold start problem with simulated user interactions" ([pre-print available here](https://arxiv.org/pdf/2503.09311)).

## Repository Structure

```
├── data/               # Data files
│   ├── candidates.csv
│   ├── candidates_reactions.csv 
│   ├── gpt_data.csv
│   ├── gpt_voters.csv
│   ├── questions.csv
│   └── results_zh.csv
├── figures/           # Generated figures
├── notebooks/         # Jupyter notebooks for analysis
│   ├── 1. Smartvote Preprocession.ipynb
│   ├── 2. GPT-4-API.ipynb
│   ├── 3. Cold-Start Dataset.ipynb
│   ├── 4. GPT Voters.ipynb
│   ├── 5. Statistical Model.ipynb
│   ├── 6. Running Simulation.ipynb
│   ├── 7. Results: Data Generation.ipynb
│   ├── 8. Results: Simulation Results.ipynb
│   └── 9. Results: Bias Investigation.ipynb
├── results/          # Generated results
└── src/             # Source code
    └── utils/       # Utility functions
```

## Requirements

- Python 3.9+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

## Data Description

- `candidates.csv`: Information about political candidates
- `candidates_reactions.csv`: Candidate responses to survey questions
- `gpt_data.csv`: GPT-4 generated responses
- `questions.csv`: Survey questions in multiple languages
- `results_zh.csv`: Election results from Zurich

## Usage

Run notebooks in order:
   - Start with data preprocessing (notebook 1)
   - Generate GPT responses (notebook 2)
   - Build cold-start dataset (notebook 3)
   - Create synthetic voters (notebook 4)
   - Train statistical model (notebook 5)
   - Run simulations (notebook 6)
   - Analyze results (notebooks 7-9)

## Citation

If you use this code in your research, please cite:

```
[Citation information will be added after publication]
```