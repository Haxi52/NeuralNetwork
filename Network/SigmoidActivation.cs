using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network
{
    public class SigmoidActivation : IActivation
    {
        public double[] Forward(double[] input)
        {
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = 1d / (1d + Math.Pow(Math.E, -input[i]));
            }
            return input;
        }
    }
}
