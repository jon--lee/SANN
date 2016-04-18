import matplotlib.pyplot as plt

filepaths = ["best_loss.log", "best_acc.log", "test_loss.log", "test_acc.log"]
blf = open('best_loss.log', 'r')
tlf = open('test_loss.log', 'r')
i = 0
iterations = []
bls = []
tls = []
prev_bl = 2.3
prev_tl = 2.3
for bl, tl in zip(blf, tlf):
    bl = float(bl)
    tl = float(tl)
    iterations.append(i)
    if bl < 1e50:
        bls.append(bl)
        prev_bl = bl
    else:
        bls.append(prev_bl)
    if tl < 1e50:
        tls.append(tl)
        prev_tl = tl
    else:
        tls.append(prev_tl)
    

    i += 1
blf.close()
tlf.close()
fig = plt.figure()
fig.suptitle("Loss", fontsize=20)
plt.plot(iterations, bls, 'g',  tls, 'r')
plt.show()

baf = open('best_acc.log', 'r')
taf = open('test_acc.log', 'r')
i = 0
iterations = []
bas = []
tas = []
for ba, ta in zip(baf, taf):
    ba = float(ba)
    ta = float(ta)
    iterations.append(i)
    bas.append(ba)
    tas.append(ta)
    i += 1
baf.close()
taf.close()
fig = plt.figure()
fig.suptitle("Accuracy", fontsize=20)
plt.plot(iterations, bas, 'g',  tas, 'r')
plt.show()


