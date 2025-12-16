# HPML Project: [Low-Latency Limit Order Book Prediction]

## Team Information

- **Team Name**: [Team Name]
- **Members**:
- Max Schettewi (mjs2430)
- Shubham Mangalvedhe
- William Wadolowski
- Nikhil Gahlot

---

## 1. Problem Statement

A Limit Order Book records all outstanding Limit Orders to buy and sell a security. These prices are continuously
updated at millisecond or sub millisecond scale

**Bid price** is defined as the price buyer is willing to pay (best bid = highest price)

**Ask price** is defined price seller is willing to accept (best ask = lowest price)

**Mid price** is the mean of best bid and best ask prices:

(best_bid + best_ask) / 2

It is interesting to predict the price movements to optimize execution strategies to know when to trade aggressively vs
passively and to automate bid/ask adjustments in market-making programs.

Transformers are a good model architecture since they model temporal dependencies and interactions across price levels
well.

---

## 2. Model Description

TODO 

---

## 3. Final Results Summary

TODO 

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### B. Wandb Dashboard
View training and evaluation metrics [here](https://wandb.ai/mjs2430-columbia-university/HPML%20LOB?nw=nwusermjs2430). 

---

### C. Specify for Training or For Inference or if Both

- **To download and prepare the dataset:**
It will prompt for a dataset location since the dataset link is one time use. Go to the provided link, click the 3 dots
and paste the url only for wget.

```bash
python scripts/load_data.py
```

- **To train the model from scratch:**
It will prompt for a wandb API key, you may opt out of wandb logging or provide a key as required.

```bash
python scripts/train.py 
```

- **To evaluate the model:**

```bash
python scripts/evaluate.py
```

- **To benchmark quantization:**

```bash
python scripts/benchmark_quantized.py
```

- **To profile the model:**

```bash
python scripts/profile_model.py
```

---

### D. Evaluation

To evaluate the trained model:

```bash
python scripts/evaluate.py
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt
# Step 2: Download dataset
python scripts/load_data.py
# Step 3: Run training (or skip if checkpoint is provided)
python scripts/train.py 
# Step 4: Evaluate
python scripts/evaluate.py
```

---

## 5. Notes

- All entry points are located in `scripts/`
- All source code is located in `src/`
- Model checkpoints are saved to `checkpoints/`
- Data is stored in `data/`
- Logs and visualizations are saved to `logs/`
- Visualization notebook is saved in `notebooks/`