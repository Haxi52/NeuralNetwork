using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    internal class TanhActivation : IActivation
    {
        public ActivationType ActivationType => ActivationType.Tanh;

        public double[] Forward(double[] input, double[] output)
        {
            for (var i =0; i < input.Length; i++)
            {
                output[i] = Math.Tanh (input[i]);
            }

            return output;
        }

        public double Derivative(double[] value, int index) => 
            1 - value[index] * value[index];
    }
}
