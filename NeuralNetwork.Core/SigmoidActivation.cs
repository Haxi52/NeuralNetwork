using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SigmoidActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.Sigmoid;
    public double[] Forward(double[] input, double[] output)
    {

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = 1d / (1d + Math.Pow(Math.E, -input[i]));
        }

        return output;
    }

    public double Derivative(double[] value, int index) => value[index] * (1.0d - value[index]);

}

