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

    public IEnumerable<ILayer> Layers => layers;

    public Network(int inputCount)
    {
        this.inputCount = inputCount;
    }

    public NetworkContext CreateContext()
    {
        var sizes = new[] { inputCount }.Concat(layers.Select(i => i.Size)).ToArray();
        return NetworkContext.Create(sizes);
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

        layers.Add(new DenseLayer(layers.Count + 1, input, size, activation));
        return this;
    }

    public double[] Process(NetworkContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));


        foreach (var layer in layers)
        {
            layer.Forward(ctx);
        }

        return ctx.Output;
    }


    public double Train(NetworkContext ctx, double rate)
    {
        ctx.Reset();

        foreach(var set in ctx.TrainingData)
        {

            var input = set.inputs;
            var expected = set.expected;

            ctx.SetInput(input);

            foreach (var layer in layers)
            {
                layer.Forward(ctx);
            }
            ctx.PushActual();

            Array.Copy(expected, ctx.Expected[ctx.Expected.Count - 1], expected.Length);
            for (var lIndex = layers.Count - 1; lIndex >= 0; lIndex--)
            {
                var layer = layers[lIndex];
                layer.Train(ctx);
            }
            ctx.Epoc();
        }

        foreach(var layer in layers)    
            layer.Apply(ctx, rate);

        return Cost(ctx);
    }


    public void Randomize()
    {
        foreach (var layer in layers)
        {
            layer.Randomize();
        }
    }


    private static double Cost(NetworkContext ctx)
    {
        var cost = 0d;
        var j = 0;
        foreach(var set in ctx.TrainingData.Select((data, i) => (data, i)))
        {
            for(var k = 0; k < ctx.Actuals[set.i].Length; k++)
            {
                cost += Math.Pow(set.data.expected[k] - ctx.Actuals[set.i][k], 2);
                j++;
            }
        }
        return cost / j;
    }
}