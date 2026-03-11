import logging
import torch

from metric import nrmse

logger = logging.getLogger(__name__)


def train(u_train, y_train, model):
    """
    Run a forward pass in teacher-forcing mode and return train NRMSE.
    """
    y_pred_train = model.forward(u_train, y_train)
    y_train = y_train[model.washout :]
    nrmse_value = nrmse(y_pred_train, y_train)
    logger.info(f"NRMSE train: {nrmse_value}")
    return float(nrmse_value)


def test(u_test, y_test, model):
    """
    Run a forward pass in autonomous mode and return test NRMSE.
    """
    y_pred_test = model.forward(u_test)
    y_test = y_test[model.washout :]
    nrmse_value = nrmse(y_pred_test, y_test)
    logger.info(f"NRMSE test: {nrmse_value}")
    return float(nrmse_value)


def test_closed_loop(dataset, model, prediction_horizon):
    """
    Closed-loop sliding window test. Returns NRMSE over all trials.

    After open-loop training, advances the reservoir through validation data
    (if present) to bridge the train->test gap, then runs closed-loop evaluation.
    """
    # Advance reservoir through validation data to prime for test start
    if hasattr(dataset, 'u_val'):
        model.forward(dataset.u_val)

    # Run closed-loop evaluation
    predictions, ground_truths = model.closed_loop_test(
        dataset.u_test, prediction_horizon
    )

    # Diagnostic: per-trial analysis in squashed domain
    n_trials = len(predictions)
    sq_errors = (predictions - ground_truths) ** 2
    diverged_mask = predictions.abs() >= 1.0
    n_diverged = diverged_mask.sum().item()

    # NRMSE in squashed domain
    nrmse_squashed = nrmse(predictions, ground_truths)
    logger.info(f"NRMSE test (squashed domain): {nrmse_squashed:.6f}")

    if n_diverged > 0:
        logger.warning(f"  {n_diverged}/{n_trials} trials diverged outside [-1,1]")
        logger.warning(f"  Diverged predictions: {predictions[diverged_mask].tolist()}")

    # Show worst 5 trials in squashed domain
    worst_idx = sq_errors.argsort(descending=True)[:5]
    for rank, idx in enumerate(worst_idx):
        logger.info(f"  Worst #{rank+1}: trial {idx.item()}, "
                     f"pred={predictions[idx]:.6f}, truth={ground_truths[idx]:.6f}, "
                     f"sq_err={sq_errors[idx]:.6f}")

    # Retransform to original coordinates: arctanh(y) + 1 (inverse of tanh(y - 1))
    # Jaeger computes NRMSE84 in the original domain (GMD Report 148, eq. 25)
    predictions_orig = torch.arctanh(torch.clamp(predictions, -0.999999, 0.999999)) + 1
    ground_truths_orig = torch.arctanh(ground_truths) + 1
    signal_variance = torch.var(torch.arctanh(dataset.u_test) + 1)
    logger.info(f"Signal variance (original domain): {signal_variance:.6f}")

    nrmse_value = nrmse(predictions_orig, ground_truths_orig, signal_variance=signal_variance)
    logger.info(f"NRMSE test (closed-loop, horizon={prediction_horizon}): {nrmse_value}")

    # NRMSE excluding diverged trials
    if n_diverged > 0 and n_diverged < n_trials:
        clean_mask = ~diverged_mask
        clean_preds = predictions_orig[clean_mask]
        clean_truths = ground_truths_orig[clean_mask]
        nrmse_clean = nrmse(clean_preds, clean_truths, signal_variance=signal_variance)
        logger.info(f"NRMSE test (excluding {n_diverged} diverged): {nrmse_clean:.6f}")

    return float(nrmse_value)


