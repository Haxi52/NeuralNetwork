using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SoftmaxActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.Softmax;
    public double[] Forward(double[] input, double[] output)
    {
        var max = .0d;
        for (var i = 0; i < output.Length; i++)
        {
            max = Math.Max(max, output[i]);
        }

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = Math.Exp(input[i] - max);
        }

        var sum = .0d;
        for (var i = 0; i < output.Length; i++)
        {
            sum += output[i];
        }

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = (output[i] / sum);
        }

        return output;
    }

    public double Derivative(double[] value, int index)
    {
        for (var i = 0; i < value.Length; i++)
            if (value[i] > value[index]) return 1d;

        return 0d;
    } 

}