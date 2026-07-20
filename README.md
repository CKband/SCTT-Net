# SCTT-Net manuscript implementation


The former Autoformer and auxiliary attention source files are retained for traceability. The revised SCTT-Net model does not import or execute them.


- The global branch is an encoder-decoder Transformer.
- The local branch is a causal dilated TCN.
- The Basic Gate uses a two-hidden-layer MLP with PCP, SW, and PET.
- The Physical Gate uses the current state and 3-day and 7-day memories.
- The learnable coefficient is `lambda = sigmoid(theta)`.
- Fusion follows `g * F_global + (1 - g) * F_local`.
- Data are split into 2002 to 2016, 2017 to 2019, and 2020 to 2022.
- Metrics include NSE, KGE, RMSE, and IOA.
- Prediction lengths of 1, 3, and 7 are supported through `PRED_LEN`.


