using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;


public interface ILayer
{
    int Size { get; }
    double[] Forward(NetworkContext ctx);
    double[] Train(NetworkContext ctx, double rate);
    void Apply(NetworkContext ctx);
    void Randomize(int? seed = null);
}