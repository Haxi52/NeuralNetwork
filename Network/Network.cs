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

    public double[] Process(ReadOnlySpan<double> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != inputCount) throw new ArgumentOutOfRangeException(nameof(inputs));


        var result = inputs.ToArray();
        foreach (var layer in layers)
        {
            result = layer.Forward(result);
        }

        return result;
    }


    public double Learn(double[] inputs, double[] expected)
    {
        var output = new double[inputs.Length];
        ProcessBatch(inputs, output);
        double cost = Cost(expected, output);
        var maxEpoc = 50;

        while (maxEpoc-- > 0)
        {
            for (var i = layers.Count - 1; i >= 0; i--)
            {
                layers[i].Evolve(cost);
            }

            ProcessBatch(inputs, output);
            double cost2 = Cost(expected, output);

            if (cost2 > cost)
            {
                maxEpoc--; // extra penaltiy if its worse
                for (var i = layers.Count - 1; i >= 0; i--)
                {
                    layers[i].Discard();
                }
            }
            else
            {
                cost = cost2;
            }
        }

        return cost;
    }

    private void ProcessBatch(double[] inputs, double[] outputs)
    {

        var batchSize = inputs.Length / Environment.ProcessorCount;
        var products = new List<(int start, int end)>();
        var i = 0;
        while (i < inputs.Length)
        {
            var start = i;
            var end = Math.Min(i + batchSize, inputs.Length);
            products.Add((start, end));
            i = end;
        }

        Parallel.ForEach(products, job =>
        {
            for (var k = job.start; k < job.end; k++)
            {
                outputs[k] = Process(new[] { inputs[k] })[0];
            }
        });
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
