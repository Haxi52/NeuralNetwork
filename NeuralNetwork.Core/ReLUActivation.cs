using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class ReLUActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.ReLU;
    public double[] Forward(double[] input, double[] output)
    {
        for (var i = 0; i < input.Length; i++)
        {
            output[i] = Math.Max(0, input[i]);
        }

        return output;
    }

    public double Activate(double input) => Math.Max(0, input);

    public double Derivative(double[] value, int index)
    {
        if (value[index] <= 0) return 0;
        return 1d;
    }
}

