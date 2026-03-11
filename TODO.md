# TODO

1. **Try different size tiles**
   Tile shape affects the periodic structure of the reservoir and directly shapes the eigenvalue spectrum. Larger tiles allow more complex internal dynamics but reduce the number of repetitions across the lattice.
   *Implementation:* Sweep `tile.shape` as a grid search parameter (e.g. `[2,2]`, `[3,3]`, `[4,4]`, `[5,5]`) and compare eigenvalue spectra, TRI distributions, and NRMSE across tile sizes. Ensure `W_args.size` is divisible by each tile dimension.

2. **Add eigenvector visualization for specific eigenvalues (magnitude and phase angle)**
   Eigenvectors reveal the spatial activation pattern of each mode — which nodes participate and how they relate in phase. Visualizing them on the lattice grid can show whether modes are localized, periodic, or distributed.
   *Implementation:* Compute eigenvectors of `W_res`, select eigenvalues of interest (e.g. dominant, purely oscillatory). Reshape each eigenvector to the `sqrt(N) x sqrt(N)` grid and plot two heatmaps per mode: one for `|v_i|` (magnitude — which nodes are active) and one for `angle(v_i)` (phase — how nodes relate temporally). Add to `tile_analysis.py` or as a standalone script.

3. **Test reservoir performance when removing specific eigenvalue modes**
   Selectively zeroing out eigenvalues and reconstructing `W_res` allows testing which modes are essential for task performance. This can reveal whether memory, oscillatory, or amplifying modes drive NRMSE.
   *Implementation:* Eigendecompose `W_res = V @ diag(λ) @ V^{-1}`. Zero out targeted eigenvalues (e.g. the half-moon cluster, purely oscillatory modes, or smallest modes), reconstruct `W_res`, and run the ESN with the modified matrix. Compare NRMSE to baseline. Could also try removing modes by magnitude range or by real/imaginary dominance.

4. **Test small tile training to full network training performance**
   Train/optimize weights on a small tile and tile it to build the full reservoir, then compare against training the full network directly. This tests whether locally optimized structure transfers effectively when repeated across the lattice.

5. **Orthogonal reservoirs**
   Use orthogonal weight matrices for the reservoir. Orthogonal matrices preserve norms and have all eigenvalues on the unit circle, providing stable dynamics without vanishing/exploding gradients and maximal memory capacity in theory.

6. **Preisach-style hysteresis activation for nanomagnet modeling**
   Replace `tanh` with a proper hysteresis operator where each node has coercivity and remanence parameters, modeling physical nanomagnet switching behavior. Unlike implicit hysteresis from high `β` + `w_self` (which is suppressed by spectral radius scaling), an explicit Preisach model gives each node an independent hysteresis loop regardless of the global SR constraint. This would make the ESN a more physically accurate model of nanomagnet lattice reservoirs.
   *Implementation:* Add a new stateful activation class in `activation.py` that tracks each node's previous output and applies asymmetric switching thresholds (different thresholds for transitioning up vs down). Parameters: coercivity (width of hysteresis loop) and remanence (output level when input is zero). Wire it up as a new `f_args.type` option in `ConfigLoader`.
