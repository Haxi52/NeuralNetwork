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

    public static Network Load(Stream stream)
    {
        using var reader = new BinaryReader(stream);
        var expectedHeader = Encoding.UTF8.GetBytes("nn");
        var header = reader.ReadBytes(2);
        var headerVersion = reader.ReadInt32();

        if (header[0] != expectedHeader[0] || header[1] != expectedHeader[1])
        {
            throw new Exception("Not a valid neural network save.");
        }
        if (headerVersion != 0xA0)
        {
            throw new Exception("Neural network save incompatible version");
        }

        var inputCount = reader.ReadInt32();

        var network = new Network(inputCount);

        while (stream.Position < stream.Length)
        {
            network.layers.Add(DenseLayer.Load(reader));
        }

        return network;
    }

    public void Save(Stream stream)
    {
        using var writer = new BinaryWriter(stream);
        writer.Write(Encoding.UTF8.GetBytes("nn"));
        writer.Write(0xA0);
        writer.Write(inputCount);

        foreach (var layer in layers)
        {
            if (layer is DenseLayer dense)
            {
                dense.Save(writer);
            }
        }

        writer.Flush();

    }

    public NetworkContext CreateContext(int? threads = null)
    {
        threads ??= Environment.ProcessorCount;
        var sizes = new[] { inputCount }.Concat(layers.Select(i => i.Size)).ToArray();
        var ctx = NetworkContext.Create(sizes);
        contextList = Enumerable.Range(0, threads.Value)
            .Select(i => NetworkContext.Create(sizes))
            .ToList();

        return ctx;
    }

    public Network AddLayer(int size, ActivationType activationType)
    {
        var input = layers.LastOrDefault()?.Size ?? inputCount;
        layers.Add(new DenseLayer(layers.Count + 1, input, size, ActivationFactory.Create(activationType)));
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


    public async Task<double> Train(NetworkContext ctx)
    {
        ctx.ShuffleTrainingData();

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
                        layer.Apply(context);
                }
            }));

        await Task.WhenAll(tasks);

        var cost = contextList.Select(i => i.GetCost()).Average();
        return cost;
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
            Array.Copy(expected, ctx.Expected[^1], expected.Length);

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