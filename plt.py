import matplotlib.pyplot as plt
import pickle

with open("./data/my_lstm_acc.pkl", "rb") as fr:
    my_acc = pickle.load(fr)
with open("./data/my_lstm_time_cost.pkl", "rb") as fr:
    my_time_cost = pickle.load(fr)
with open("./data/lstm_acc.pkl", "rb") as fr:
    acc = pickle.load(fr)
with open("./data/lstm_time_cost.pkl", "rb") as fr:
    time_cost = pickle.load(fr)

x = list(range(len(my_acc)))
plt.figure()
plt.plot(x, my_acc, color="green", label="my lstm")
plt.plot(x, acc, color="red", label="offical lstm")
plt.title("accuracy")
plt.legend()

plt.figure()
plt.plot(x, my_time_cost, color="green", label="my lstm")
plt.plot(x, time_cost, color="red", label="offical lstm")
plt.title("time cost")
plt.legend()
plt.show()
