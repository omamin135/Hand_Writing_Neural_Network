import train


datas = train.loadMNISTData("mnist.pkl.gz")

testingData = list(datas[2])

net = train.NeuralNet([784,30,10])
net.testData("Trained_Data/784_30'_10.npz", testingData)
