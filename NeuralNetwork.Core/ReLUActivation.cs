using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class ReLUActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.ReLU;
    public double[] Forward(NetworkContext ctx, int index)
    {
        var input = ctx.PreOutput[index];
        var output = ctx.LayerOutput[index];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = Math.Max(0, input[i]);
        }

        return output;
    }

    public double Activate(double input) => Math.Max(0, input);

    public double Prime(double input) => input;
}

