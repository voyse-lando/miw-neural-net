﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace miw_neural_net
{
    public static class XORNOR
    {
        public static Action<string> Output = (_) => { };

        public static void Start()
        {
            double[][] data = InputParser.TableFromFile("./xor_nor.txt")!;

            double[][] inputs = [
                [data[0][0], data[0][1]],
                [data[1][0], data[1][1]],
                [data[2][0], data[2][1]],
                [data[3][0], data[3][1]]
            ];

            double[][] outputs = [
                [data[0][2], data[0][3]],
                [data[1][2], data[1][3]],
                [data[2][2], data[2][3]],
                [data[3][2], data[3][3]]
            ];

            // Tworzenie i trening sieci
            NeuralNetwork nn = new NeuralNetwork(learningRate: 0.3, beta: 1.3, inputSize: 2, layers: [2, 2, 2]);
            //NeuralNetwork nn = new NeuralNetwork(learningRate: 0.3f, beta: 1.0f);
            Output("Rozpoczęcie treningu...");
            nn.Train(inputs, outputs, epochs: 200000);
            Output("Trening zakończony.");

            // Testowanie sieci
            Output("\nTestowanie sieci:");
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = nn.Predict(inputs[i]);
                Output($"Wejście: [{inputs[i][0]}, {inputs[i][1]}], " +
                                 $"Oczekiwane: [{outputs[i][0]}, {outputs[i][1]}], " +
                                 $"Otrzymane: [{prediction[0]:F2}, {prediction[1]:F2}], " +
                                 $"Błąd: [{Math.Abs(outputs[i][0] - prediction[0]):F2}, {Math.Abs(outputs[i][1] - prediction[1]):F2}]");
            }
        }
    }
}
