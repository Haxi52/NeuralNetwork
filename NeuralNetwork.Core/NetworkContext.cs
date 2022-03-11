﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public class NetworkContext
    {
        private int trainingEpocs = 0;
        private Stack<double[]> actualsCache = new Stack<double[]>();

        internal List<double[]> PreOutput { get; } = new();
        internal List<double[]> LayerOutput { get; } = new();
        internal List<double[]> Expected { get; } = new();
        internal List<double[]> Actuals { get; } = new();
        internal List<double[]> AdjustedWeights { get; } = new();
        internal List<double[]> AdjustedBiases { get; } = new();
        internal int TrainingEpocs => trainingEpocs;

        public double[] Input => LayerOutput[0];
        public double[] Output => LayerOutput.Last();

        public List<(double[] inputs, double[] expected)> TrainingData { get; } = new();

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

        public void SetInput(double[] inputs)
        {
            if (inputs == null || inputs.Length != Input.Length)
                throw new ArgumentException(nameof(inputs));
            lock (this)
            {
                Array.Copy(inputs, Input, inputs.Length);
            }
        }

        internal void Reset()
        {
            foreach (var item in Actuals)
                actualsCache.Push(item);

            Actuals.Clear();

            foreach (var biases in AdjustedBiases)
            {
                Array.Clear(biases);
            }
            foreach (var weights in AdjustedWeights)
            {
                Array.Clear(weights);
            }

            trainingEpocs = 0;
        }

        internal void Epoc() => Interlocked.Increment(ref trainingEpocs);

        internal void PushActual()
        {
            lock (this)
            {
                if (!actualsCache.TryPop(out var actual))
                {
                    actual = new double[Output.Length];
                }
                Array.Copy(Output, actual, Output.Length);
                Actuals.Add(actual);
            }
        }
    }
}
