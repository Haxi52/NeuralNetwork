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
        private double[]? lastOutput;

        public int Size { get; }
        public int InputSize { get; }

        public DenseLayer(int inputs, int size)
        {
            Size = size;
            InputSize = inputs;

            biases = new double[size];
            weights = new double[size * inputs];
        }

        public double[] Forward(Span<double> input)
        {
            var output = new double[Size];
            Array.Copy(biases, output, Size);

            var i = 0;
            for (var k = 0; k < input.Length; k++)
            {
                for (var j = 0; j < InputSize; j++)
                {
                    output[j] += weights[i++] * input[k];
                }
            }

            lastOutput = output;
            return output;
        }

        public double[] Backward(Span<double> expected, double learningRate)
        {
            if (expected.Length != Size) throw new ArgumentException(nameof(expected));
            if (lastOutput is null) throw new InvalidOperationException("Cannot go backwards before you go forwards");

            var z = new double[Size];

            for (var k = 0; k < Size; k++)
            {
                z[k] = lastOutput[k] - expected[k];
            }

            var dw = new double[Size];
            var i = 0;
            for (var j = 0; j < InputSize; j++)
            {
                for (int k = 0; k < Size; k++)
                {
                    dw[k] += z[k] * weights[i++];
                }
            }

            i = 0;
            for (var j = 0; j < InputSize; j++)
            {
                for (int k = 0; k < Size; k++)
                {
                    dw[k] += z[k] * weights[i++];
                }
            }

            //for (int k = 0; k < Size; k++)
            //{
            //    dw[k] += dw[k] * (1 / InputSize) * learningRate;
            //}

            return z;
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
    }
}
