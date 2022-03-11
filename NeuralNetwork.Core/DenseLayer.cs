﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class DenseLayer : ILayer
{
    private readonly double[] biases;
    private readonly double[] weights;

    private readonly int index;
    private readonly IActivation activation;

    public int Size { get; }
    public int InputSize { get; }

    public DenseLayer(int index, int inputs, int size, IActivation activation)
    {
        Size = size;
        this.activation = activation;
        this.index = index;
        InputSize = inputs;

        biases = new double[size];
        weights = new double[size * inputs];
    }

    public double[] Forward(NetworkContext ctx)
    {
        var input = ctx.LayerOutput[index - 1];
        var output = ctx.PreOutput[index];
        Array.Copy(biases, output, Size);

        var i = 0;
        for (var j = 0; j < Size; j++)
        {
            for (var k = 0; k < input.Length; k++)
            {
                output[j] += weights[i++] * input[k];
            }
        }

        return activation.Forward(ctx, index);
    }

    public void Randomize(int? seed = null)
    {
        var rng = new Random(seed ?? (int)DateTime.Now.Ticks);
        var range = Size / 2;
        for (var i = 0; i < biases.Length; i++)
        {
            biases[i] = (rng.NextDouble() * range) - (range / 2.0d);
        }

        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] = (rng.NextDouble() * range) - (range / 2.0d);
        }
    }


    public double[] Train(NetworkContext ctx) // how fast to move the weights/biases to improve the cost
    {
        var input = ctx.LayerOutput[index - 1];
        var output = ctx.Expected[index - 1];
        var expected = ctx.Expected[index];
        Array.Copy(input, output, output.Length);

        var i = 0;

        for (var j = 0; j < Size; j++) // for each neuron in this layer
        {
            var actual = ctx.LayerOutput[index][j];
            var error = actual - expected[j];

            ctx.AdjustedBiases[index][j] += error;
            for (var k = 0; k < output.Length; k++) // for each input neuron
            {
                output[k] += weights[i] * error * ctx.PreOutput[index][j];

                ctx.AdjustedWeights[index][i] += error * input[k];//  * weights[i];
                i++;
            }
        }

        return output;
    }

    public void Apply(NetworkContext ctx, double rate)
    {

        for (var i = 0; i < Size; i++)
        {
            biases[i] -= (ctx.AdjustedBiases[index][i] / ctx.TrainingEpocs) * rate;
        }

        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] -= (ctx.AdjustedWeights[index][i] / ctx.TrainingEpocs) * rate;
        }
    }

    public override string ToString()
    {
        var output = new StringBuilder();
        var w = 0;
        for (var n = 0; n < Size; n++)
        {
            for (var j = 0; j < InputSize; j++)
            {
                output.Append($"|w{n},{j}({weights[w]:0.000})|");
                w++;
            }
        }
        output.AppendLine();
        for (var n = 0; n < Size; n++)
        {
            output.Append($"|b{n}({biases[n]:0.000})|");
        }

        return output.ToString();
    }
}
