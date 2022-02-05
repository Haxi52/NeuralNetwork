using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class NoActivation : IActivation
{
    public double[] Forward(double[] input)
    {
        return input;
    }
}
