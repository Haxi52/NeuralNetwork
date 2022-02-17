using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class ReLUActivation : IActivation
{
    public double[] Forward(NetworkContext ctx, int index)
    {
        var input = ctx.LayerOutput[index];
        var output = ctx.LayerActivated[index];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = Math.Max(0, input[i]);
        }
        return output;
    }

    public double Activate(double input) => Math.Max(0, input);
}

