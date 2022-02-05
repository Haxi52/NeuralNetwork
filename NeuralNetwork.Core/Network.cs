using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

public class Network
{
    private readonly List<ILayer> layers = new();
    private readonly int inputCount;

    public Network(int inputCount)
    {
        this.inputCount = inputCount;
    }

    public Network AddLayer(int size, ActivationType activationType)
    {
        var input = layers.LastOrDefault()?.Size ?? inputCount;

        IActivation activation = activationType switch
        {
            ActivationType.None => new NoActivation(),
            ActivationType.ReLU => new ReLUActivation(),
            ActivationType.Softmax => new SoftmaxActivation(),
            ActivationType.Sigmoid => new SigmoidActivation(),
            _ => throw new Exception(),
        };

        layers.Add(new DenseLayer(input, size, activation));
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


    public double Learn(double[][] inputSet, double[][] expectedSet, double rate)
    {
        var cache = new List<double[]>();
        var cost = 0d;

        for (var set = 0; set < inputSet.Length; set++) // foreach training instance
        {
            var input = inputSet[set];
            var expected = expectedSet[set];    

            for (var i = 0; i < input.Length; i++) 
            {

                var foward = Pool.Instance.Borrow(inputCount);
                Array.Copy(input, foward, inputCount);

                foreach (var layer in layers)
                {
                    cache.Add(foward);
                    foward = layer.Forward(foward);
                }

                var learn = Pool.Instance.Borrow(layers.Last().Size);
                Array.Copy(expected, learn, expected.Length);

                if (cost == 0d)
                    cost = Cost(learn, foward);
                else
                    cost = (cost + Cost(learn, foward)) * 0.5d;

                Pool.Instance.Return(foward);

                for (var lIndex = layers.Count - 1; lIndex >= 0; lIndex--)
                {
                    var layer = layers[lIndex];
                    var actual = cache.Last();
                    cache.Remove(actual);
                    learn = layer.Learn(actual, learn, rate);
                    Pool.Instance.Return(actual);
                }
                Pool.Instance.Return(learn);
            }
        }

        return cost;
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

/*
 * n * w = o
 * 
 * n * (w - e) = y
 * 
 * w - e = y / n
 * 
 * (y / n) - w = -e
 * 
 * 
 */