import Perceptron from "./class/Perceptron.js";
import {shuffle} from "./utils/utils.js";

const myPerceptron = new Perceptron(2)
const dataset = [
    [[20, 5], [-1]],
    [[18, 25], [-1]],
    [[24, 15], [-1]],
    [[22, 25], [-1]],
    [[21, 0], [-1]],
    [[25, 15], [-1]],

    [[5, 45], [1]],
    [[9, 5], [1]],
    [[12, 15], [1]],
    [[7, 25], [1]],
    [[8, 75], [1]],
    [[15, 75], [1]],
]

console.log('before training prediction')
dataset.forEach(data => {
    console.log(`For ${data[0]} expected : ${data[1][0]} | obtain : ${myPerceptron.predict(data[0])}`)
})
console.log(myPerceptron.weights)

for(let i = 0; i<500; i++) {
    const ds = shuffle(dataset)
    ds.forEach(data => {
        myPerceptron.train(data[0], data[1][0])
    })
}

console.log(myPerceptron.weights)
console.log('after training prediction')
dataset.forEach(data => {
    console.log(`For ${data[0]} expected : ${data[1][0]} | obtain : ${myPerceptron.predict(data[0])}`)
})
