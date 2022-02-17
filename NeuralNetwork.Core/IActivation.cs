using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

public interface IActivation
{
    public double[] Forward(NetworkContext ctx, int index);
    public double Activate(double input);
}

public enum ActivationType
{
    None = 0,
    ReLU,
    Softmax,
    Sigmoid,
}

