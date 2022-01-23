using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network;

internal class Network
{
    private readonly List<ILayer> layers = new List<ILayer>();
    private readonly int inputCount;
    private double[]? lastOutput;

    public Network(int inputCount)
    {
        this.inputCount = inputCount;
    }

    public Network AddDenseLayer(int size)
    {
        var input = layers.LastOrDefault()?.Size ?? inputCount;

        layers.Add(new DenseLayer(input, size));
        return this;
    }

    public Network AddActivationLayer(ActivationType activationType)
    {
        layers.Add(new ActivationLayer(layers.Last().Size, activationType));
        return this;
    }

    public double[] Process(Span<double> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != inputCount) throw new ArgumentOutOfRangeException(nameof(inputs));


        var result = inputs.ToArray();
        foreach (var layer in layers)
        {
            result = layer.Forward(result);
        }

        lastOutput = result;
        return result;
    }

    public double Cost(Span<double> expected)
    {
        if (lastOutput == null) throw new InvalidOperationException("Cannot calculate cost without running network");
        if (expected == null) throw new ArgumentNullException(nameof(expected));

        double result = 0d;
        for(var i = 0; i < expected.Length; i++)
        {
            result += Math.Pow(expected[i] - lastOutput[i], 2);
        }

        return result / expected.Length;
    }

    public void Learn(Span<double> expected, double learningRate)
    {
        double[] result = expected.ToArray();
        for(var i = layers.Count - 1; i >= 0; i--)
        {
            result = layers[i].Backward(result, learningRate);
        }
    }

    public void Randomize()
    {
        foreach (var layer in layers)
        {
            layer.Randomize();
        }
    }

    public static double CCELoss(int category, double[] input)
    {

        return -(double)Math.Log(Math.Clamp(input[category], 1e-7, 1 - 1e-7));
    }

    public static double MeanSquareErrorLoss(double[] trueValues, double[] predictedValues)
    {
        return (double)trueValues.Zip(predictedValues).Select(v => Math.Pow(v.First - v.Second, 2)).Average();

    }
}
