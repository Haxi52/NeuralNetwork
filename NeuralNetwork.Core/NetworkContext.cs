﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public class NetworkContext
    {
        private Stack<double[]> actualsCache = new Stack<double[]>();


        internal List<double[]> LayerOutput { get; } = new();
        internal List<double[]> LayerActivated { get; } = new();
        internal List<double[]> Expected { get; } = new();
        internal List<double[]> Actuals { get; } = new();

        public double[] Input => LayerOutput[0];
        public double[] Output => LayerOutput.Last();

        public List<(double[] inputs, double[] expected)> TrainingData { get; } = new();

        private NetworkContext() {}
        internal static NetworkContext Create(params int[] sizes)
        {
            var ctx = new NetworkContext(); 
            foreach(var size in sizes)
            {
                ctx.LayerActivated.Add(new double[size]);
                ctx.LayerOutput.Add(new double[size]);
                ctx.Expected.Add(new double[size]);
            }
            return ctx;
        }

        public void SetInput(double[] inputs)
        {
            if (inputs == null || inputs.Length != Input.Length)
                throw new ArgumentException(nameof(inputs));

            Array.Copy(inputs, Input, inputs.Length);
        }

        internal void Reset()
        {
            foreach(var item in Actuals)
                actualsCache.Push(item);

            Actuals.Clear();
        }

        internal void PushActual()
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