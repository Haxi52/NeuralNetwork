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
            biases[i] = rng.NextDouble() - 0.5d;
        }

        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] = (rng.NextDouble() * 8) - 4.0d;
        }
    }



    public double[] Learn(NetworkContext ctx,
                          double rate) // how fast to move the weights/biases to improve the cost
    {
        var input = ctx.LayerActivated[index - 1];
        var output = ctx.Expected[index - 1];
        Array.Clear(output);

        var m = (1.0d / InputSize) * rate;
        var i = 0;
        for (var j = 0; j < Size; j++) // for each neuron in this layer
        {
            var actual = ctx.LayerActivated[index][j];
            var deltaCost = (actual - ctx.Expected[index][j]);// * actual * (1 - actual);
                        
            biases[j] -= deltaCost * m;
            for (var k = 0; k < input.Length; k++) // for each input neuron
            {
                var delta = deltaCost * input[k];
                weights[i] -= delta * m;
                output[k] += weights[i] * deltaCost * m;
                i++;
            }
        }

        return output;
    }


    //public double[] Learn(double[] input, // input from previous layer
    //                      double[] expected, // expected output given the inputs, 
    //                      double rate) // how fast to move the weights/biases to improve the cost
    //{
    //    var output = Pool.Instance.Borrow(InputSize);
    //    var m = (1.0d / InputSize) * rate;

    //    var i = 0;
    //    for (var j = 0; j < Size; j++) // for each neuron
    //    {
    //        var sumActual = 0d;
    //        for (var k = 0; k < input.Length; k++) // for each weight
    //        {
    //            var actual = activation.Activate(weights[i] * input[k]);
    //            var delta = actual - expected[j];
    //            weights[i] -= delta * m;

    //            output[k] += weights[i] * expected[j];
    //            sumActual += delta;
    //            i++;
    //        }
    //        // biases[j] -= sumActual * m;

    //    }

    //    Pool.Instance.Return(expected);

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
