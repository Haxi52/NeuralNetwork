using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SigmoidActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.Sigmoid;
    public double[] Forward(NetworkContext ctx, int index)
    {
        var input = ctx.PreOutput[index];
        var output = ctx.LayerOutput[index];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = 1d / (1d + Math.Pow(Math.E, -input[i]));
        }

        return output;
    }

    public double Prime(double input) => input * (1.0d - input);
}

