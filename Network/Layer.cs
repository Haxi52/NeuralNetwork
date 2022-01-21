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

internal class Layer
{
    private readonly double[] biases;
    private readonly double[] weights;

    private double[]? output;

    public int Size { get; }
    public ActivationType ActivationType { get; }
    public int InputSize { get; }

    public Layer(int inputs, int size, ActivationType activationType)
    {
        Size = size;
        ActivationType = activationType;
        InputSize = inputs;

        this.biases = new double[size];
        this.weights = new double[size * inputs];
    }


    public double[] Forward(Span<double> input)
    {
        output = new double[Size];
        Array.Copy(biases, output, Size);

        for (var s = 0; s < input.Length; s++)
        {
            var min = (Size * s);
            var max = (Size * (s + 1));
            var j = 0;
            for (var i = min; i < max; i++)
            {
                output[j++] += weights[i] * input[s];
            }
        }

        return ActivationType switch
        {
            ActivationType.Softmax => Activate_Softmax(output),
            ActivationType.ReLU => Activate_ReLU(output),
            ActivationType.None => output,
            _ => throw new InvalidOperationException("Unknown activation type")
        };
    }

    private double[] Activate_ReLU(double[] input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = Math.Max(0, input[i]);
        }
        return input;
    }

    private double[] Activate_Softmax(double[] input)
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

internal struct LayerProperties
{
    public int Size;
    public ActivationType ActivationType;
}
