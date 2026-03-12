def qlike_loss(y_true, y_pred):
    return torch.mean(y_true / y_pred - torch.log(y_true / y_pred) - 1)
