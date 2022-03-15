using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

internal class NoActivation : IActivation
{
    public ActivationType ActivationType => ActivationType.None;

    public double[] Forward(NetworkContext ctx, int index)
    {
        Array.Copy(ctx.PreOutput[index], ctx.LayerOutput[index], ctx.PreOutput[index].Length);
        return ctx.LayerOutput[index];
    }

    public double Activate(double input) => input;
    public double Prime(double input) => input;
}
