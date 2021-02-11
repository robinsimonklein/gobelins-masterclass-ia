import PerceptronLite from "./PerceptronLite.js";

export default class DenseNet {
    hiddenLayer1 = []
    hiddenOutput = []
    outputLayer = []
    nbInput = 0

    /**
     * @param {number} nbInput - Number of input
     * @param {number} nbHiddenNeuron - Number of hidden neuron
     * @param {number} nbOfOutput - Number of output
     */
    constructor(nbInput, nbHiddenNeuron, nbOfOutput) {
        this.nbInput = nbInput

        for(let i = 0; i < nbHiddenNeuron; i++) {
            this.hiddenLayer1.push(new PerceptronLite(nbInput))
        }

        for(let i = 0; i < nbOfOutput; i++) {
            this.outputLayer.push(new PerceptronLite(nbHiddenNeuron))
        }
    }

    /**
     * Predict
     * @param {[number]} input
     * @returns {[number]}
     */
    predict(input) {
        for(let i = 0; i < this.hiddenLayer1.length; i++) {
            this.hiddenOutput[i] = this.hiddenLayer1[i].predict(input)
        }

        let networkOutput = this.outputLayer[0].predict(this.hiddenOutput)

        return [networkOutput]
    }

    /**
     * Calculate network error
     * @param {[number]} output
     * @param {[number]} expected
     * @returns {[number]}
     */
    calcNetworkError(output, expected) {
        const outputErrors = []
        for(let i = 0; i<output.length; i++) {
            outputErrors.push(output[i] - expected[i])
        }
        return outputErrors
    }

    /**
     * Train
     * @param {[number]} input
     * @param {[number]} expected
     * @param {number} lr - learningRate
     */
    train(input, expected, lr) {

        const prediction = this.predict(input)
        const netError = this.calcNetworkError(prediction, expected)

        let outDelta = this.outputLayer.map(() => 0.0)

        // Pour chacune des sorties du réseau
        for(let j = 0; j < this.outputLayer.length; j++) {
            // Calculer terme à terme la somme des erreurs multipliées par
            // la dérrivée de chaque neurone caché
            for(let i = 0; i < this.hiddenLayer1.length; i++) {
                outDelta[j] += this.hiddenLayer1[i].dActivate() * netError[j]
            }
        }

        // Calcul du gradient en fonciton du delta pour
        // chaque neurone de sortie
        for(let i=0; i<outDelta.length; i++) {
            this.outputLayer[i].getGradient(outDelta[i])
        }

        // Créer un tableau d'erreurs de taille nbHiddenNeuron
        // (donc la taille de hiddenLayer1) ne contenant que des 0
        const l1Errors = this.hiddenLayer1.map(() => 0.0)

        // Obtenir l'erreur de chaque neurone caché
        let counter = 0
        this.outputLayer.forEach(pOut => {
            for(let i=0; i<this.hiddenLayer1.length; i++) {
                l1Errors[i] += pOut.weights[i] * outDelta[counter]
            }
            counter += 1
        })

        // Obtenir le delta de chaque neurone caché
        const hiddenDelta = []
        for(let i=0; i<this.hiddenLayer1.length; i++) {
            hiddenDelta[i] = l1Errors[i] * this.hiddenLayer1[i].dActivate()
        }

        for(let i=0; i<hiddenDelta.length; i++) {
            this.hiddenLayer1[i].getGradient(hiddenDelta[i])
        }

        this.updateWeights(lr)
    }

    /**
     * Update weights
     * @param {number} lr - learning rate
     */
    updateWeights(lr) {
        this.outputLayer.forEach(perceptron => {
            perceptron.updateWeights(lr)
        })
        this.hiddenLayer1.forEach(perceptron => {
            perceptron.updateWeights(lr)
        })
    }

}
