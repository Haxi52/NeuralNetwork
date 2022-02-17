using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class NoActivation : IActivation
{
    public double[] Forward(NetworkContext ctx, int index)
    {
        Array.Copy(ctx.LayerOutput[index], ctx.LayerActivated[index], ctx.LayerOutput[index].Length);
        return ctx.LayerActivated[index];
    }

    public double Activate(double input) => input;
}
