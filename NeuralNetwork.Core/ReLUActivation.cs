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
        var data = ctx.LayerOutput[index];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = Math.Max(0, data[i]);
        }

        return data;
    }

    public double Activate(double input) => Math.Max(0, input);

    public double Prime(double input) => input;
}

