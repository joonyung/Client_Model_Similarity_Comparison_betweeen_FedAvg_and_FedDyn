import matplotlib.pyplot as plt



layers = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
avg_iid = [0.5498,0.6077,0.6373,0.6612,0.6722,0.6873,0.6912,0.6939,0.6923,0.7064,0.7042,0.7098,0.7104,0.7136,0.715,0.7156,0.7201,0.7175,0.7192,0.7228]
avg_noniid = [0.3445,0.4323,0.4998,0.5282,0.5408,0.5646,0.5786,0.5875,0.595,0.6239,0.6148,0.6296,0.6263,0.6365,0.6326,0.6407,0.6425,0.6379,0.6469,0.6418]
dyn_iid = [0.4457,0.5763,0.5931,0.6282,0.6333,0.6437,0.6528,0.6541,0.6606,0.661,0.6627,0.6659,0.6728,0.6798,0.6756,0.6795,0.6863,0.6953,0.6917,0.6943]
dyn_noniid = [0.335,0.4966,0.5319,0.5315,0.5352,0.5699,0.5537,0.5605,0.5963,0.5773,0.5962,0.5903,0.5846,0.5497,0.5871,0.5932,0.6241,0.5754,0.598,0.6017]


plt.plot(layers, avg_iid, label = "FedAvg(IID)", marker = 'o', color = 'coral')
plt.plot(layers, avg_noniid, label = "FedAvg(nonIID)", marker = 'o', color = 'darksalmon')
plt.plot(layers, dyn_iid, label = "FedDyn(IID)", marker = 'o', color = 'cornflowerblue')
plt.plot(layers, dyn_noniid, label = "FedDyn(nonIID)", marker = 'o', color = 'royalblue')

plt.legend(loc=4)

plt.xlabel("Communication Rounds")
plt.ylabel("Test Accuracy")

plt.axis([-5, 105, 0.25, 0.85])
# plt.grid(True)

plt.title('Learning Curves for FedAvg and FedDyn')

plt.show()

plt.savefig('./trained_models/accuracy')
