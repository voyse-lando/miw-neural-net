using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace miw_neural_net
{
    public class NeuralNetwork
    {
        public static Action<string> Output = (_) => { };

        private readonly double learningRate;
        private readonly double beta;
        private readonly int inputSize;
        private readonly int[] layers;
        private double[][][] weights = default!;

        public NeuralNetwork(double learningRate, double beta, int inputSize, int[] layers)
        {
            this.learningRate = learningRate;
            this.beta = beta;
            this.inputSize = inputSize;
            this.layers = layers;

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rand = new Random();
            weights = new double[layers.Length][][];

            weights[0] = new double[layers[0]][];


            // Input
            for (int i = 0; i < layers[0]; ++i)
            {
                weights[0][i] = new double[inputSize + 1]; // +1 bias
                for (int k = 0; k < weights[0][i].Length; ++k)
                {
                    weights[0][i][k] = rand.NextDouble() * 10 - 5;
                }
            }

            // Hidden
            for (int lr = 1; lr < layers.Length; ++lr)
            {
                weights[lr] = new double[layers[lr - 1]][];
                for (int i = 0; i < layers[lr]; ++i)
                {
                    weights[lr][i] = new double[layers[lr - 1] + 1]; // +1 bias
                    for (int k = 0; k < weights[lr][i].Length; ++k)
                    {
                        weights[lr][i][k] = rand.NextDouble() * 10 - 5;
                    }
                }
            }
        }
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-beta * x));

        private double SigmoidDerivative(double x) => beta * x * (1 - x);

        public double[] Predict(double[] inputs)
        {
            double[] currentOutputs = inputs;

            for (int lr = 0; lr < layers.Length; ++lr)
            {
                double[] nextOutputs = new double[layers[lr]];
                for (int i = 0; i < layers[lr]; ++i)
                {
                    double sum = weights[lr][i][0]; // bias
                    for (int k = 0; k < currentOutputs.Length; ++k)
                    {
                        sum += weights[lr][i][k + 1] * currentOutputs[k];
                    }
                    nextOutputs[i] = Sigmoid(sum);
                }
                currentOutputs = nextOutputs;
            }

            return currentOutputs;
        }

        public void Train(double[][] trainingInputs, double[][] trainingOutputs, int epochs)
        {
            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                var shuffledIndices = Enumerable.Range(0, trainingInputs.Length).OrderBy(x => Guid.NewGuid()).ToArray();

                foreach (var index in shuffledIndices)
                {
                    double[] inputs = trainingInputs[index];
                    double[] desiredOutputs = trainingOutputs[index];

                    double[][] layerOutputs = new double[layers.Length + 1][];
                    layerOutputs[0] = inputs;

                    for (int lr = 0; lr < layers.Length; ++lr)
                    {
                        layerOutputs[lr + 1] = new double[layers[lr]];
                        for (int i = 0; i < layers[lr]; ++i)
                        {
                            double sum = weights[lr][i][0]; // bias
                            for (int k = 0; k < layerOutputs[lr].Length; ++k)
                            {
                                sum += weights[lr][i][k + 1] * layerOutputs[lr][k];
                            }
                            layerOutputs[lr + 1][i] = Sigmoid(sum);
                        }
                    }

                    double[][] deltas = new double[layers.Length][];
                    int lastLayer = layers.Length - 1;

                    deltas[lastLayer] = new double[layers[lastLayer]];
                    for (int i = 0; i < layers[lastLayer]; ++i)
                    {
                        double error = desiredOutputs[i] - layerOutputs[lastLayer + 1][i];
                        deltas[lastLayer][i] = learningRate * error * SigmoidDerivative(layerOutputs[lastLayer + 1][i]);
                    }
                    for (int lr = layers.Length - 2; lr >= 0; lr--)
                    {
                        deltas[lr] = new double[layers[lr]];
                        for (int i = 0; i <= layers[lr] - 1; ++i)
                        {
                            double error = 0;
                            for (int k = 0; k < layers[lr + 1]; ++k)
                            {
                                error += deltas[lr+1][k] * weights[lr + 1][k][i + 1]; // +1 skip bias
                            }
                            deltas[lr][i] = error * SigmoidDerivative(layerOutputs[lr + 1][i]);
                        }
                    }

                    for (int lr = 0; lr < layers.Length; ++lr)
                    {
                        for (int i = 0; i < layers[lr]; ++i)
                        {
                            weights[lr][i][0] += deltas[lr][i]; // bias

                            for (int k = 0; k < layerOutputs[lr].Length; ++k)
                            {
                                weights[lr][i][k + 1] += deltas[lr][i] * layerOutputs[lr][k];
                            }
                        }
                    }
                }


                if (epoch % 1000 == 0)
                {
                    double totalError = 0;
                    for (int i = 0; i < trainingInputs.Length; ++i)
                    {
                        double[] output = Predict(trainingInputs[i]);
                        for (int k = 0; k < output.Length; ++k)
                        {
                            totalError += Math.Abs(trainingOutputs[i][k] - output[k]);
                        }
                    }
                    Output($"Epoka {epoch}: Średni błąd = {totalError / (trainingInputs.Length * trainingOutputs[0].Length):F4}");
                }
            }
        }
    }

}
