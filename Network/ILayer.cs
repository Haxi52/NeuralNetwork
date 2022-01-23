using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network;


internal interface ILayer
{
    int Size { get; }
    double[] Forward(Span<double> input);
    double[] Backward(Span<double> expected, double learningRate);
    void Randomize(int? seed = null);
}