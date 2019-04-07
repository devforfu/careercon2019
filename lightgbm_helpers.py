def accuracy(y_true, y_pred):
    n = len(y_true)
    y_hat = y_pred.reshape(9, n).argmax(axis=0)
    value = (y_true == y_hat).mean()
    return 'accuracy', value, True
