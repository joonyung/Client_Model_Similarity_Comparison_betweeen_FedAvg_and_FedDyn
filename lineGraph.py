import matplotlib.pyplot as plt



layers = [1, 2, 3, 4, 5, 6, 7]
avg_iid = [0.988266, 0.981869667, 0.965248667, 0.928366, 0.891135, 0.819415333, 0.880003]
avg_noniid = [0.983799333, 0.977401, 0.95197, 0.888247667, 0.789285, 0.740761667, 0.788229667]
dyn_iid = [0.991744667, 0.987159667, 0.977378667, 0.9488, 0.915858333, 0.852850333, 0.899424]
dyn_noniid = [0.992799, 0.986654333, 0.975976, 0.947435333, 0.860242333, 0.858567333, 0.866840667]


# plt.plot(layers, avg_iid, label = "FedAvg", marker = 'o', color = 'coral')
plt.plot(layers, avg_noniid, label = "FedAvg", marker = 'o', color = 'coral')
# plt.plot(layers, dyn_iid, label = "FedDyn", marker = 'o', color = 'cornflowerblue')
plt.plot(layers, dyn_noniid, label = "FedDyn", marker = 'o', color = 'cornflowerblue')

plt.legend(loc=3)

plt.xlabel("Layer")
plt.ylabel("CKA Similarity")

plt.axis([0.8, 7.2, 0.675, 1.01])
plt.grid(True)

plt.title('CKA Similarity (nonIID)')

plt.show()

plt.savefig('./trained_models/cka_noniid')
