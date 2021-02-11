export default class Perceptron {
    weights = []

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
        let sop = 0.0;

        for(let i = 0; i < input.length; i++) {
            sop += input[i] * this.weights[i]
        }

        sop += this.bias

        return this.activate(sop)
    }

    /**
     * Activate
     * @param {number} x
     * @returns {number}
     */
    activate(x) {
        return x > 0 ? 1 : -1
    }

    /**
     * Calculate error
     * @param {number} output
     * @param {number} expected
     * @returns {number}
     */
    calcError(output, expected) {
        return expected - output
    }

    /**
     * Train function
     * @param {[number]} input - Input
     * @param {number} expected - Expected
     * @param {number} learningRate - Learning rate
     */
    train(input, expected, learningRate = 0.4) {
        // On regarde quelle est la prédiction
        let netOutput = this.predict(input)

        // On vérifie si la prédiction est juste par rapport
        // à la réponse qu'on attendait
        let netError = this.calcError(netOutput, expected)

        for(let i = 0; i < input.length; i++) {
            this.weights[i] = this.weights[i] + (netError * input[i]) * learningRate
        }

        this.bias += netError * learningRate
    }
}
