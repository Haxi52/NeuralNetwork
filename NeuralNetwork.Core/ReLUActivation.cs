using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class ReLUActivation : IActivation
{
    public double[] Forward(double[] input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = Math.Max(0, input[i]);
        }
        return input;
    }

    public double Activate(double input) => Math.Max(0, input);
}

