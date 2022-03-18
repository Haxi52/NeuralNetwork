using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class NoActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.None;

    public double[] Forward(double[] input, double[] output)
    {
        Array.Copy(input, output, input.Length);
        return output;
    }

    public double Derivative(double[] value, int index) => 1d;
}
