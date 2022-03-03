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
        var data = ctx.LayerOutput[index];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = Math.Exp(data[i]);
        }

        var sum = .0d;
        for (var i = 0; i < data.Length; i++)
        {
            sum += data[i];
        }

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = (data[i] / sum);
        }

        return data;
    }

    public double Activate(double input) => input;

    public double Prime(double input) => input;
}