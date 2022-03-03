using System;
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
        var output = ctx.LayerOutput[index];
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
        for (var i = 0; i < biases.Length; i++)
        {
            biases[i] =  (rng.NextDouble() * 4d) - 2.0d;
        }

        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] = (rng.NextDouble() * 4d) - 2.0d;
        }
    }


    public double[] Train(NetworkContext ctx,
                          double rate) // how fast to move the weights/biases to improve the cost
    {
        var input = ctx.LayerOutput[index - 1];
        var output = ctx.Expected[index - 1];
        var expected = ctx.Expected[index];
        Array.Clear(output);

        var m = (1.0d / InputSize) * rate;
        var i = 0;

        for (var j = 0; j < Size; j++) // for each neuron in this layer
        {
            var actual = ctx.LayerOutput[index][j];
            var deltaCost = actual - expected[j];
            var deltaZ = actual * activation.Prime(actual);
            var error = (deltaCost * deltaZ) + double.Epsilon;

            // ctx.AdjustedBiases[index][j] = (ctx.AdjustedBiases[index][j] + error) * 0.5d;
            biases[j] -= error * m;
            for (var k = 0; k < output.Length; k++) // for each input neuron
            {
                output[k] += weights[i] * expected[j];
                weights[i] -= error * input[k] * m;
                // ctx.AdjustedWeights[index][i] = (ctx.AdjustedWeights[index][i] + (error * input[k])) * 0.5d;
                i++;
            }
        }

        return output;
    }

    public void Apply(NetworkContext ctx)
    {
        for(var i = 0; i < Size; i++)
        {
            biases[i] -= ctx.AdjustedBiases[index][i];
        }

        for(var i = 0; i < weights.Length; i ++)
        {
            weights[i] -= ctx.AdjustedWeights[index][i];
        }
    }



    //public double[] Learn(NetworkContext ctx,
    //                      double rate) // how fast to move the weights/biases to improve the cost
    //{
    //    var input = ctx.LayerOutput[index - 1];
    //    var output = ctx.Expected[index - 1];
    //    var expected = ctx.Expected[index];
    //    Array.Clear(output);

    //    var m = (1.0d / InputSize) * rate;
    //    var i = 0;
    //    for (var j = 0; j < Size; j++) // for each neuron in this layer
    //    {
    //        var actual = ctx.LayerOutput[index][j];
    //        var error = (actual - expected[j]) * activation.Prime(actual);

    //        biases[j] -= error * m;
    //        for (var k = 0; k < output.Length; k++) // for each input neuron
    //        {
    //            // var delta = error * input[k];

    //            output[k] += weights[i] * expected[j];
    //            weights[i] -= error * input[k] * m;
    //            i++;
    //        }
    //    }

    //    return output;
    //}


    public override string ToString()
    {
        var output = new StringBuilder();
        var w = 0;
        for(var n = 0; n < Size; n++)
        {
            for (var j =0; j < InputSize; j++)
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
