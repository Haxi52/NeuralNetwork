using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SoftmaxActivation : IActivation
{
    public double[] Forward(NetworkContext ctx, int index)
    {
        var input = ctx.PreOutput[index];
        var output = ctx.LayerOutput[index];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = Math.Exp(input[i]);
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

    public double Activate(double input) => input;

    public double Prime(double input) => input;
}