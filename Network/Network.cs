using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network;

internal class Network
{
    private readonly List<Layer> layers = new List<Layer>();
    private readonly int inputSize;
    private readonly int outputSize;

    public Network(params LayerProperties[] layerProperties)
    {
        inputSize = layerProperties[0].Size;

        var prevSize = inputSize;
        foreach (var layer in layerProperties.Skip(1))
        {
            layers.Add(new Layer(prevSize, layer.Size, layer.ActivationType));
            prevSize = layer.Size;
        }
        outputSize = prevSize;
    }

    public double[] Process(Span<double> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != inputSize) throw new ArgumentOutOfRangeException(nameof(inputs));

        var result = inputs.ToArray();
        foreach (var layer in layers)
        {
            result = layer.Forward(result);
        }

        return result;
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
