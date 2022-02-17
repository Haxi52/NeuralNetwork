using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class SigmoidActivation : IActivation
{
    public double[] Forward(NetworkContext ctx, int index)
    {
        var input = ctx.LayerOutput[index];
        var output = ctx.LayerActivated[index];

        for (var i = 0; i < input.Length; i++)
        {
            output[i] = 1d / (1d + Math.Pow(Math.E, -input[i]));
        }
        return output;
    }
    public double Activate(double input) => 1d / (1d + Math.Pow(Math.E, -input));
}

