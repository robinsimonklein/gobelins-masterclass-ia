export default class PerceptronLite {
    weights = []
    input = []
    gradients = []
    sop = 0
    bias = 0
    act = 0

    /**
     * @param {number} nbInputs - Number of inputs
     */
    constructor(nbInputs) {
        for(let i = 0; i < nbInputs; i++) {
            this.weights.push(Math.random())
        }
        this.bias = Math.random()
    }

    /**
     * Fonction de prediction
     * @param {[number]} input
     * @returns {number}
     */
    predict(input = []) {
        this.input = input

        this.sop = 0.0

        for(let i = 0; i < this.input.length; i++) {
            this.sop += this.input[i] * this.weights[i]
        }

        this.sop += this.bias
        this.act = this.activate(this.sop)

        return this.act
    }

    /**
     * Activate
     * @param {number} x
     * @returns {number}
     */
    activate(x) {
        // Fonction d'activation ReLU
        // Equivaut à x < 0 ? 0 : x
        return Math.max(0, x)
    }

    /**
     * Derivative activate
     * @returns {number}
     */
    dActivate() {
        return this.sop > 0 ? 1 : 0
    }

    /**
     * Get gradient
     * @param {number} delta
     * @returns {[number]}
     */
    getGradient(delta) {
        this.gradients = []
        for(let i = 0; i < this.input.length; i++) {
            this.gradients.push(this.input[i] * delta)
        }
    }

    /**
     * Update weights
     * @param {number} lr - learning rate
     */
    updateWeights(lr) {
        for(let i=0; i<this.weights.length; i++) {
            this.weights[i] -= this.gradients[i] * lr
        }
    }
}
