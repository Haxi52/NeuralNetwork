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
        var data = ctx.LayerOutput[index];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = 1d / (1d + Math.Pow(Math.E, -data[i]));
        }

        return data;
    }
    public double Activate(double input) => 1d / (1d + Math.Pow(Math.E, -input));

    //public double Prime(double input) => Math.Pow(Math.E, -input) / Math.Pow(1 + Math.Pow(Math.E, -input), 2); 
    public double Prime(double input) => input * (1 - input);
}

