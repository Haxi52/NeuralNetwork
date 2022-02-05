using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;


public interface ILayer
{
    int Size { get; }
    double[] Forward(double[] input);
    double[] Learn(double[] input, double[] expected, double rate);
    void Randomize(int? seed = null);
}