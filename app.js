import DenseNet from "./class/DenseNet.js";
import {shuffle} from "./utils/utils.js";

const datasetAND = [
    [[1.0, 0.0], [0.0]],
    [[0.0, 1.0], [0.0]],
    [[1.0, 1.0], [1.0]],
    [[0.0, 0.0], [0.0]],
]

const net = new DenseNet(2, 4, 1)

datasetAND.forEach(d => {
    console.log(`For ${d[0]} | expect ${d[1]} | get ${net.predict(d[0])}`)
})

console.log('Out weight')
console.log(net.outputLayer.map(l => l.weights)[0])
console.log('Hidden weight')
console.log(net.hiddenLayer1.map(l => l.weights))

console.log('________________________')
for(let i=0; i<900; i++) {
    const datasetANDshuffle = shuffle(datasetAND)
    datasetANDshuffle.forEach(d => {
        net.train(d[0], d[1], 0.01)
    })
}
console.log('---------TRAIN----------')
console.log('________________________')

console.log('Out weight')
console.log(net.outputLayer.map(l => l.weights)[0])
console.log('Hidden weight')
console.log(net.hiddenLayer1.map(l => l.weights))

datasetAND.forEach(d => {
    console.log(`For ${d[0]} | expect ${d[1]} | get ${net.predict(d[0])}`)
})
