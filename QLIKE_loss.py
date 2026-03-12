def qlike_loss(y_true, y_pred):
    # Đảm bảo y_pred luôn dương để tránh lỗi log
    return torch.mean(y_true / y_pred - torch.log(y_true / y_pred) - 1)
