using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core;

public interface IActivation
{
    public ActivationType ActivationType { get; }
    public double[] Forward(double[] input, double[] output);
    public double Derivative(double[] value, int index);

}

public enum ActivationType
{
    None = 0,
    ReLU,
    Softmax,
    Sigmoid,
    Tanh,
}

