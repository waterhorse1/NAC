# NAC

Official PyTorch implementation of NAC from the paper: <font face = "BlackBold">Neural Auto-Curricula in Two-Player Zero-Sum Games</font>.

We offer trained models and test code for 2D-RPS visualisation and Kuhn Poker->Leduc Poker Generalization test.
## How to run
Firstly create and activate the required conda environment.
```python
conda env create -f lmac.yml
conda activate lmac
```

### 2D-RPS Testing
![123](2d_rps.png)
Visualisation results can be tested in 2d_rps/visualization_2d_rps.ipynb.

### Kuhn->Leduc Generalization
![123](kuhn_leduc_gen.png)

we provide a local implementation in which one can reproduce the results of generalising our models trained on Kuhn Poker to Leduc Poker.

```python
cd leduc_poker
# To reproduce the approximate best-response results
python3 kuhn_to_leduc.py --br_type 'approx_br_rand'
# To reproduce the exact best-response results
python3 kuhn_to_leduc.py --br_type 'exact_br'
```



