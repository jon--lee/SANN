filepaths = ["best_loss.log", "best_acc.log", "test_loss.log", "test_acc.log"]
for path in filepaths:
    f = open(path, 'w')
    f.close()
