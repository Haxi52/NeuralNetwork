// See https://aka.ms/new-console-template for more information



using NeuralNetwork.Core;

Console.WriteLine("Hello, World!");

var generations = 0;
var network = new Network(2);
network.AddLayer(2, ActivationType.Sigmoid);
network.Randomize();


var inputSet = new[]
{
    new[] { 0d, 0d },
    new[] { 1d, 0d },
    new[] { 0d, 1d },
    new[] { 1d, 1d },
};

var expectedSet = new[]
{
    new[] { 1d, 1d, },
    new[] { 0d, 1d, },
    new[] { 1d, 0d, },
    new[] { 0d, 0d, },
};

var result = network.Process(inputSet[0]);
var cost = 0d;
Console.WriteLine($"input: {string.Join(", ", inputSet[0].Select(i => $"{i:0.0000}"))}");
Console.WriteLine($"first: {string.Join(", ", result.Select(i => $"{i:0.0000}"))}");

for (var k = 0; k < 100; k++)
{
    for (var i = 0; i < 10000; i++)
    {
        cost = network.Learn(inputSet, expectedSet, 0.00001d);
        generations++;
    }
    var result2 = network.Process(inputSet[0]);
    Console.WriteLine($"{k}: {string.Join(", ", result2.Select(i => $"{i:0.0000}"))}    |   cost: {cost}    |   gen: {generations}");
}