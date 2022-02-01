using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network;

internal class Network
{
    private readonly List<ILayer> layers = new();
    private readonly int inputCount;

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

    public double[] Process(double[] inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != inputCount) throw new ArgumentOutOfRangeException(nameof(inputs));


        var result = inputs;
        foreach (var layer in layers)
        {
            var input = result;
            result = layer.Forward(input);
            Pool.Instance.Return(input);
        }

        return result;
    }


    public double Learn(double[] input, double[] expected, double rate)
    {
        var cache = new List<double[]>();

        for (var i = 0; i < input.Length; i++)
        {

            var result = Pool.Instance.Borrow(1);
            result[0] = input[i];

            foreach (var layer in layers)
            {
                cache.Add(result);
                result = layer.Forward(result);
            }
            Pool.Instance.Return(result);

            var e = Pool.Instance.Borrow(1); 
            e[0] = expected[i];
            foreach (var layer in layers)
            {
                result = cache.Last();
                cache.Remove(result);
                e = layer.Learn(e, result, rate);
                Pool.Instance.Return(result);
            }
            Pool.Instance.Return(e);
            cache.Clear();
        }

        return 0d;
    }


    public void Randomize()
    {
        foreach (var layer in layers)
        {
            layer.Randomize();
        }
    }


    private static double Cost(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual)
    {
        if (expected == null) throw new ArgumentNullException(nameof(expected));
        if (actual == null) throw new ArgumentNullException(nameof(actual));

        double result = 0d;
        for (var i = 0; i < expected.Length; i++)
        {
            result += Math.Pow(expected[i] - actual[i], 2);
        }

        return result / expected.Length;
    }
}
