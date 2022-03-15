using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    internal class ActivationFactory
    {
        public static IActivation Create(ActivationType activationType)
        {
            return activationType switch
            {
                ActivationType.None => new NoActivation(),
                ActivationType.ReLU => new ReLUActivation(),
                ActivationType.Softmax => new SoftmaxActivation(),
                ActivationType.Sigmoid => new SigmoidActivation(),
                _ => throw new Exception(),
            };
        }
    }
}
