
def clear():
    filepaths = ["logs/best_loss_sann.log", "logs/best_acc_sann.log", "logs/test_loss_sann.log", "logs/test_acc_sann.log",
            "logs/best_loss_hc.log", "logs/best_acc_hc.log", "logs/test_loss_hc.log", "logs/test_acc_hc.log"]
    for path in filepaths:
        f = open(path, 'w')
        f.close()



