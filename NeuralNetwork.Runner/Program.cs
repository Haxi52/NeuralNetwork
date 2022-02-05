// See https://aka.ms/new-console-template for more information



using NeuralNetwork.Core;

Console.WriteLine("Hello, World!");

var generations = 0;
var network = new Network(2);
network.AddLayer(2, ActivationType.Sigmoid);
network.Randomize();


var input = new[] { 0d, 0d };
var expected = new[] { 1d, 0d, };

var result = network.Process(input);

var cost = network.Learn(input, expected, 0.0005d);
generations++;



Console.WriteLine(string.Join(", ", input.Select(i => $"{i:0.0000}")));
Console.WriteLine(string.Join(", ", result.Select(i => $"{i:0.0000}")));