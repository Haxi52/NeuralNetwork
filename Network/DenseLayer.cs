using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network
{
    internal class DenseLayer : ILayer
    {
        private const double evolutionRate = 0.001d;
        
        private readonly double[] biases;
        private readonly double[] weights;
        private readonly double[] evolutionWeightsCache;
        private readonly double[] evolutionBiasesCache;

        public int Size { get; }
        public int InputSize { get; }

        public DenseLayer(int inputs, int size)
        {
            Size = size;
            InputSize = inputs;

            biases = new double[size];
            weights = new double[size * inputs];

            evolutionBiasesCache = new double[biases.Length];
            evolutionWeightsCache = new double[weights.Length];
        }

        public double[] Forward(Span<double> input)
        {
            var output = new double[Size];
            Array.Copy(biases, output, Size);

            var i = 0;
            for (var k = 0; k < input.Length; k++)
            {
                for (var j = 0; j < Size; j++)
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

        public void Evolve(double cost)
        {
            Array.Copy(weights, evolutionWeightsCache, weights.Length); 
            Array.Copy(biases, evolutionBiasesCache, biases.Length);

            var rng = new Random((int)DateTime.Now.Ticks);
            var localRate = evolutionRate;

            for (var i = 0; i < biases.Length; i++)
            {
                biases[i] += (rng.NextDouble() * localRate) - (localRate * 0.5d);
            }

            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] += (rng.NextDouble() * localRate) - (localRate * 0.5d);
            }
        }

        public void Discard()
        {
            Array.Copy(evolutionWeightsCache, weights, weights.Length);
            Array.Copy(evolutionBiasesCache, biases, biases.Length);
        }
    }
}
