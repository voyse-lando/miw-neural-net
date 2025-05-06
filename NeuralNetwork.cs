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

        private readonly float learningRate;
        private readonly float beta;
        private float[][] hiddenWeights;
        private float[] outputWeights;

        public NeuralNetwork(float learningRate = 0.3f, float beta = 1.0f)
        {
            this.learningRate = learningRate;
            this.beta = beta;
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rand = new Random();

            hiddenWeights = new float[2][];
            for (int i = 0; i < 2; i++)
            {
                hiddenWeights[i] = new float[3];
                for (int j = 0; j < 3; j++)
                {
                    hiddenWeights[i][j] = (float)(rand.NextDouble() * 10 - 5);
                }
            }

            outputWeights = new float[3];
            for (int i = 0; i < 3; i++)
            {
                outputWeights[i] = (float)(rand.NextDouble() * 10 - 5);
            }
        }

        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-beta * x));
        }

        private float SigmoidDerivative(float x)
        {
            return beta * x * (1 - x);
        }

        public float Predict(float[] inputs)
        {
            float[] hiddenOutputs = new float[2];
            for (int i = 0; i < 2; i++)
            {
                float sum = hiddenWeights[i][0];
                sum += hiddenWeights[i][1] * inputs[0];
                sum += hiddenWeights[i][2] * inputs[1];
                hiddenOutputs[i] = Sigmoid(sum);
            }

            float outputSum = outputWeights[0];
            outputSum += outputWeights[1] * hiddenOutputs[0];
            outputSum += outputWeights[2] * hiddenOutputs[1];

            return Sigmoid(outputSum);
        }

        public void Train(float[][] inputs, float[] outputs, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var shuffledIndices = Enumerable.Range(0, inputs.Length).OrderBy(x => Guid.NewGuid()).ToList();

                foreach (int index in shuffledIndices)
                {
                    float[] input = inputs[index];
                    float target = outputs[index];

                    float[] hiddenSums = new float[2];
                    float[] hiddenOutputs = new float[2];

                    for (int i = 0; i < 2; i++)
                    {
                        hiddenSums[i] = hiddenWeights[i][0];
                        hiddenSums[i] += hiddenWeights[i][1] * input[0];
                        hiddenSums[i] += hiddenWeights[i][2] * input[1];
                        hiddenOutputs[i] = Sigmoid(hiddenSums[i]);
                    }

                    float outputSum = outputWeights[0];
                    outputSum += outputWeights[1] * hiddenOutputs[0];
                    outputSum += outputWeights[2] * hiddenOutputs[1];
                    float output = Sigmoid(outputSum);

                    float outputError = (target - output) * SigmoidDerivative(output);

                    float[] hiddenErrors = new float[2];
                    for (int i = 0; i < 2; i++)
                    {
                        hiddenErrors[i] = outputError * outputWeights[i + 1] * SigmoidDerivative(hiddenOutputs[i]);
                    }

                    outputWeights[0] += learningRate * outputError;
                    outputWeights[1] += learningRate * outputError * hiddenOutputs[0];
                    outputWeights[2] += learningRate * outputError * hiddenOutputs[1];

                    for (int i = 0; i < 2; i++)
                    {
                        hiddenWeights[i][0] += learningRate * hiddenErrors[i];
                        hiddenWeights[i][1] += learningRate * hiddenErrors[i] * input[0];
                        hiddenWeights[i][2] += learningRate * hiddenErrors[i] * input[1];
                    }

                    float newOutput = Predict(input);
                    float newError = Math.Abs(target - newOutput);
                }

                if (epoch % 1000 == 0)
                {
                    float totalError = 0;
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        float output = Predict(inputs[i]);
                        totalError += Math.Abs(outputs[i] - output);
                    }
                    Output($"Epoka {epoch}, Średni błąd: {totalError / inputs.Length:F4}");
                }
            }
        }
    }
}
