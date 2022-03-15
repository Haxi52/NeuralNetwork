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
    private List<NetworkContext> contextList = new();

    public IEnumerable<ILayer> Layers => layers;

    public Network(int inputCount)
    {
        this.inputCount = inputCount;
    }

    public NetworkContext CreateContext(int threads)
    {
        var sizes = new[] { inputCount }.Concat(layers.Select(i => i.Size)).ToArray();
        var ctx = NetworkContext.Create(sizes);
        contextList = Enumerable.Range(0, threads) 
            .Select(i => NetworkContext.Create(sizes))
            .ToList();

        return ctx;
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


    public async Task<double> Train(NetworkContext ctx, double rate)
    {
        var tasks = contextList
            .Select(context =>
            Task.Run(() =>
            {
                ctx.CopyTo(context);
                context.Reset();
                TrainInternal(context, 0, contextList.Count);

                lock (ctx)
                {
                    foreach (var layer in layers)
                        layer.Apply(context, rate);
                }
            }));

        await Task.WhenAll(tasks);

        return contextList.Select(i => i.GetCost()).Average();  
    }

    private void TrainInternal(NetworkContext ctx, int partition, int scale)
    {
        for (var i = partition; i < ctx.TrainingData.Count; i += scale)
        {
            var input = ctx.TrainingData[i].inputs;
            var expected = ctx.TrainingData[i].expected;
            var actual = ctx.TrainingData[i].actual;

            ctx.SetInput(input);

            foreach (var layer in layers)
            {
                layer.Forward(ctx);
            }

            Array.Copy(ctx.Output, actual, ctx.Output.Length);
            Array.Copy(expected, ctx.Expected[ctx.Expected.Count - 1], expected.Length);

            for (var lIndex = layers.Count - 1; lIndex >= 0; lIndex--)
            {
                var layer = layers[lIndex];
                layer.Train(ctx);
            }
            ctx.Epoc();
        }
    }


    public void Randomize()
    {
        foreach (var layer in layers)
        {
            layer.Randomize();
        }
    }

}