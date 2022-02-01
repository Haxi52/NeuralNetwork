using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer
{
    public class Pool
    {
        private readonly static Pool instance = new Pool();
        public static Pool Instance => instance;
        private Pool() { }

        private readonly ConcurrentDictionary<int, Stack<double[]>> pool = new();

        public int Count => pool[16]?.Count ?? 0;

        public double[] Borrow(int size)
        {
            var queue = pool.GetOrAdd(size, i => new Stack<double[]>());

            if (queue.TryPop(out var value)) return value;
            return new double[size];
        }

        public void Return(double[] obj)
        {
            var queue = pool.GetOrAdd(obj.Length, i => new Stack<double[]>());
            queue.Push(obj);
        }
    }
}
