﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network
{
    internal class DenseLayer : ILayer
    {
        private readonly double[] biases;
        private readonly double[] weights;
        private readonly IActivation activation;

        public int Size { get; }
        public int InputSize { get; }

        public DenseLayer(int inputs, int size, IActivation activation)
        {
            Size = size;
            this.activation = activation;
            InputSize = inputs;

            biases = new double[size];
            weights = new double[size * inputs];
        }

        public double[] Forward(double[] input)
        {
            var output = Pool.Instance.Borrow(Size);
            Array.Copy(biases, output, Size);

            var i = 0;
            for (var j = 0; j < Size; j++)
            {
                for (var k = 0; k < input.Length; k++)
                {
                    output[j] += weights[i++] * input[k];
                }
            }

            return activation.Forward(output);
        }

        public void Randomize(int? seed = null)
        {
            var rng = new Random(seed ?? (int)DateTime.Now.Ticks);
            for (var i = 0; i < biases.Length; i++)
            {
                biases[i] = rng.NextDouble() - 0.5d;
            }

            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] = (rng.NextDouble() * 8) - 4.0d;
            }
        }


        public double[] Learn(double[] input, // input from previous layer
                              double[] expected, // expected output given the inputs, 
                              double rate) // how fast to move the weights/biases to improve the cost
        {
            var actual = Forward(input);
            var output = Pool.Instance.Borrow(InputSize);

            var i = 0;
            for (var j = 0; j < Size; j++) // for each neuron
            {
                var dCost_dActual = actual[j] - expected[j];   
                var dActual_dZ = actual[j] * (1 - actual[j]);
                var nError = dCost_dActual * dActual_dZ;

                biases[j] += nError * rate;
                for (var k = 0; k < input.Length; k++) // for each weight
                {
                    var delta = nError * input[k];  
                    weights[i] += delta * rate;

                    output[k] += weights[i] * expected[j];
                    i++;
                }

            }

            Pool.Instance.Return(expected);
            Pool.Instance.Return(actual);

            return activation.Forward(output);
        }
    }
}
