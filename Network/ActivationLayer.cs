using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network;

internal enum ActivationType
{
    None,
    ReLU,
    Softmax,
}

internal class ActivationLayer : ILayer
{
    private readonly ActivationType activationType;
    private double[]? lastOutput;
    public int Size { get; }

    public ActivationLayer(int size, ActivationType activationType)
    {
        Size = size;
        this.activationType = activationType;
    }

    public double[] Forward(double[] input)
    {
        var output = new double[input.Length];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = input[i];
        }

        lastOutput = activationType switch
        {
            ActivationType.Softmax => Activate_Softmax(output),
            ActivationType.ReLU => Activate_ReLU(output),
            ActivationType.None => output,
            _ => throw new InvalidOperationException("Unknown activation type")
        };

        return lastOutput;
    }


    private static double[] Activate_ReLU(double[] input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = Math.Max(0, input[i]);
        }
        return input;
    }

    private static double[] Activate_Softmax(double[] input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = Math.Exp(input[i]);
        }

        var sum = .0d;
        for (var i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }

        for (var i = 0; i < input.Length; i++)
        {
            input[i] = (input[i] / sum);
        }

        return input;
    }

    public void Randomize(int? seed = null) { }


    public double[] Learn(double[] input, double[] expected, double rate)
    {
        return input;
    }
}
