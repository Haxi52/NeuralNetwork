// See https://aka.ms/new-console-template for more information



using NeuralNetwork.Core;
using System.Diagnostics;

Console.WriteLine("Hello, World!");

var generations = 0;
var network = new Network(2);
network.AddLayer(2, ActivationType.Sigmoid);
network.AddLayer(2, ActivationType.Sigmoid);
network.Randomize();

var ctx = network.CreateContext();

var inputSet = new[]
{
    //new[] { 0d, 0d },
    new[] { 1d, 0d },
    new[] { 0d, 1d },
    //new[] { 1d, 1d },
};

var expectedSet = new[]
{
    //new[] { 1d, 1d, },
    new[] { 0d, 1d, },
    new[] { 1d, 0d, },
   // new[] { 0d, 0d, },
};

ctx.TrainingData.Clear();
for (var i = 0; i < inputSet.Length; i++)
{
    ctx.TrainingData.Add((inputSet[i], expectedSet[i]));
}

ctx.SetInput(inputSet[0]);
var result = network.Process(ctx);
// var cost = 0d;
Console.WriteLine($"input: {string.Join(", ", inputSet[0].Select(i => $"{i:0.0000}"))}");
Console.WriteLine($"expec: {string.Join(", ", expectedSet[0].Select(i => $"{i:0.0000}"))}");
Console.WriteLine($"first: {string.Join(", ", result.Select(i => $"{i:0.0000}"))}");

Console.WriteLine();

await LearnALot();
// LoopInteractive();


void LoopInteractive()
{

    foreach (var layer in network.Layers)
    {
        Console.Write(layer.ToString());
    }
    Console.WriteLine();

    while (Console.ReadKey().Key != ConsoleKey.Escape)
    {
        var cost = network.Train(ctx, 0.05d);
        generations++;
        ctx.SetInput(inputSet[0]);
        var result2 = network.Process(ctx);
        Console.WriteLine($"after: {string.Join(", ", result2.Select(i => $"{i:0.0000}"))} | cost: {cost:0.000000} | gen: {generations}");

        foreach (var layer in network.Layers)
        {
            Console.Write(layer.ToString());
        }
        Console.WriteLine();
    }

}

async Task LearnALot()
{
    var cost = 0d;
    var sw = new Stopwatch();
    sw.Start();

    for (var k = 0; k < 100; k++)
    {
        for (var i = 0; i < 1000; i++)
        {
            cost = await network.Train(ctx, 0.05d);
            generations++;
        }
        ctx.SetInput(inputSet[0]);
        var result2 = network.Process(ctx);
        Console.WriteLine($"after: {string.Join(", ", result2.Select(i => $"{i:0.0000}"))} | cost: {cost:0.000000} | gen: {generations}");
    }

    sw.Stop();
    Console.WriteLine($"Completed in {sw.Elapsed}");
}