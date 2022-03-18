using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public class NetworkContext
    {
        private int trainingEpocs = 0;
        private double minLearningRate = 0;
        private double maxLearningRate = 0;

        internal List<double[]> PreOutput { get; } = new();
        internal List<double[]> LayerOutput { get; } = new();
        internal List<double[]> Expected { get; } = new();
        internal List<double[]> AdjustedWeights { get; } = new();
        internal List<double[]> AdjustedBiases { get; } = new();
        internal int TrainingEpocs => trainingEpocs;
        internal double LearningRate { get; private set; }

        public double[] Input => LayerOutput[0];
        public double[] Output => LayerOutput.Last();
        public List<(double[] inputs, double[] expected, double[] actual)> TrainingData { get; } = new();

        private NetworkContext() { }

        internal static NetworkContext Create(params int[] sizes)
        {
            var ctx = new NetworkContext();
            var prevSize = 0;
            foreach (var size in sizes)
            {
                ctx.PreOutput.Add(new double[size]);
                ctx.LayerOutput.Add(new double[size]);
                ctx.Expected.Add(new double[size]);
                ctx.AdjustedBiases.Add(new double[size]);
                ctx.AdjustedWeights.Add(new double[size * prevSize]);

                prevSize = size;
            }
            return ctx;
        }

        internal void ShuffleTrainingData()
        {
            var rng = new Random();
            int n = TrainingData.Count;
            while (n > 1)
            {
                int k = rng.Next(n--);
                (TrainingData[k], TrainingData[n]) = (TrainingData[n], TrainingData[k]);
            }
        }

        internal NetworkContext CopyTo(NetworkContext other)
        {
            other.LearningRate = LearningRate;

            if (TrainingData.Count != other.TrainingData.Count)
            {
                other.TrainingData.Clear();
                foreach (var (inputs, expected, actual) in TrainingData)
                {
                    other.TrainingData.Add((new double[inputs.Length], new double[expected.Length], new double[actual.Length]));
                }

                for (var i = 0; i < TrainingData.Count; i++)
                {
                    Array.Copy(TrainingData[i].inputs, other.TrainingData[i].inputs, TrainingData[i].inputs.Length);
                    Array.Copy(TrainingData[i].expected, other.TrainingData[i].expected, TrainingData[i].expected.Length);
                    Array.Copy(TrainingData[i].actual, other.TrainingData[i].actual, TrainingData[i].actual.Length);
                }
            }

            return other;
        }

        public void SetLearningRate(double rate)
        {
            LearningRate = rate;
        }

        public void SetInput(double[] inputs)
        {
            if (inputs == null || inputs.Length != Input.Length)
                throw new ArgumentException(null, nameof(inputs));
            Array.Copy(inputs, Input, inputs.Length);
        }

        internal void Reset()
        {
            foreach (var biases in AdjustedBiases)
            {
                Array.Clear(biases);
            }
            foreach (var weights in AdjustedWeights)
            {
                Array.Clear(weights);
            }
            foreach (var (inputs, expected, actual) in TrainingData)
            {
                Array.Fill(actual, double.NaN);
            }
            trainingEpocs = 0;
        }

        internal void Epoc() => Interlocked.Increment(ref trainingEpocs);

        internal double GetCost()
        {
            var cost = 0d;
            var j = 0;
            foreach (var set in TrainingData.Select((data, i) => (data, i)))
            {
                for (var k = 0; k < set.data.actual.Length; k++)
                {
                    if (double.IsNaN(set.data.actual[k])) continue;
                    cost += Math.Pow(set.data.expected[k] - set.data.actual[k], 2);
                }
                j += set.data.actual.Length;
            }
            return cost / j;
        }

    }
}
