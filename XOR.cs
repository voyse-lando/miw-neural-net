using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace miw_neural_net
{
    public static class XOR
    {
        public static Action<string> Output = (_) => { };

        public static void Start()
        {
            double[][] inputs = new double[][]
            {
                new double[] {0, 0},
                new double[] {0, 1},
                new double[] {1, 0},
                new double[] {1, 1}
            };

            double[][] outputs = [[0], [1], [1], [0]];

            // Tworzenie i trening sieci
            NeuralNetwork nn = new NeuralNetwork(learningRate: 0.3, beta: 1.0, inputSize: 2, layers: [2, 1]);
            //NeuralNetwork nn = new NeuralNetwork(learningRate: 0.3f, beta: 1.0f);
            Output("Rozpoczęcie treningu...");
            nn.Train(inputs, outputs, epochs: 50000);
            Output("Trening zakończony.");

            // Testowanie sieci
            Output("\nTestowanie sieci:");
            for (int i = 0; i < inputs.Length; i++)
            {
                double prediction = nn.Predict(inputs[i])[0];
                Output($"Wejście: [{inputs[i][0]}, {inputs[i][1]}], " +
                                 $"Oczekiwane: {outputs[i]}, " +
                                 $"Otrzymane: {prediction:F2}, " +
                                 $"Błąd: {Math.Abs(outputs[i][0] - prediction):F2}");
            }
        }
    }
}
