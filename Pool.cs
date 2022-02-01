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

        private readonly ConcurrentDictionary<int, Queue<double[]>> pool = new();
        private readonly ConcurrentBag<double[]> borrowed = new();

        public double[] Borrow(int size)
        {
            var queue = pool.GetOrAdd(size, i => new Queue<double[]>());

            if (queue.TryDequeue(out var value)) return value;
            var result = new double[size];
            borrowed.Add(result);
            return result;
        }

        public void Return(double[] obj)
        {
            var queue = pool.GetOrAdd(obj.Length, i => new Queue<double[]>());
            queue.Enqueue(obj);
        }
    }
}
