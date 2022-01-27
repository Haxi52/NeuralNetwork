using System;
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

        public int Size { get; }
        public int InputSize { get; }

        public DenseLayer(int inputs, int size)
        {
            Size = size;
            InputSize = inputs;

            biases = new double[size];
            weights = new double[size * inputs];
        }

        public double[] Forward(double[] input)
        {
            var output = new double[Size];
            Array.Copy(biases, output, Size);

            var i = 0;
            for (var j = 0; j < Size; j++)
            {
                for (var k = 0; k < input.Length; k++)
                {
                    output[j] += weights[i++] * input[k];
                }
            }

            return output;
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

        public double[] Learn(double[] input, double[] expected, double rate)
        {
            var output = new double[InputSize];
            var m = 1 / InputSize;

            var i = 0;
            for (var j = 0; j < Size; j++)
            {
                var sumActual = 0d;
                for (var k = 0; k < input.Length; k++)
                {
                    var actual = weights[i] * input[k];
                    var delta = actual - expected[j];
                    weights[i] -= delta * rate * m;

                    output[k] += weights[i] * expected[j];
                    sumActual += delta;
                    i++;
                }

                biases[j] -= sumActual * rate * m;  
            }

            return output;
        }
    }
}
