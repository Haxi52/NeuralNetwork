﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer.Network
{
    public class ReLUActivation : IActivation
    {
        public double[] Forward(double[] input)
        {
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = Math.Max(0, input[i]);
            }
            return input;
        }
    }
}
