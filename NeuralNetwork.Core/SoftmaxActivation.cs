using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SoftmaxActivation : IActivation
{
    public double[] Forward(double[] input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = Math.Exp(input[i]);
        }

        var sum = .0d;
        for (var i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }

        for (var i = 0; i < input.Length; i++)
        {
            input[i] = (input[i] / sum);
        }

        return input;
    }

    public double Activate(double input) => input;
}