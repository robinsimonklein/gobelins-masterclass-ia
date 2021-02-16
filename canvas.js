const canvas = document.querySelector('#canvas')
const selectNumber = document.querySelector('#number')
const addBtn = document.querySelector('#add')
const trainBtn = document.querySelector('#train')
const predictBtn = document.querySelector('#predict')
let ctx = canvas.getContext('2d')

const penSize = 10
const inputSize = 20
let points = []
let dataset = []

canvas.addEventListener('mousemove', (e) => {
    if (e.buttons !== 1) return;

    let circle = new Path2D();
    circle.moveTo(125, 35);
    const x = e.offsetX;
    const y = e.offsetY;

    points.push([x,y])

    circle.arc(x, y, penSize, 0, 2 * Math.PI);

    ctx.fill(circle);

})

addBtn.addEventListener('click', () => {
    if(points.length < 1) {
        console.error('No draw')
        return
    }

    points = points.slice(0, inputSize)
    console.log(points)

    dataset.push([[...points], [parseInt(selectNumber.value)]])

    console.log(dataset)

    // cleat canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = []
})
trainBtn.addEventListener('click', () => {
    console.log('train')
})
predictBtn.addEventListener('click', () => {
    console.log('predict')
})
