import logging

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


