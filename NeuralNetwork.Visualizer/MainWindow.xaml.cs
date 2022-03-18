using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NeuralNetwork.Core;

namespace NeuralNetworkVisualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly double graphSize = 2d;

        private readonly List<UIElement> GraphPoints = new List<UIElement>();
        private readonly List<UIElement> ExpectedPoints = new List<UIElement>();
        private readonly List<UIElement> PredictedPoints = new List<UIElement>();

        private Network network;
        private NetworkContext ctx;
        private int generations = 0;
        private double cost = 0f;
        private readonly double minLearningRate = 0.001d;
        private readonly double maxLearningRate = 0.1d;
        private volatile bool isLearning = false;

        public MainWindow()
        {
            InitializeComponent();

            this.Title = "Neural Network Visualizer";

            Canvas.Loaded += (sender, e) =>
            {
                DrawGraph();
                DrawExpected();
            };

            network = new Network(1)
                .AddLayer(32, ActivationType.ReLU)
                .AddLayer(32, ActivationType.Softmax)
                //.AddLayer(32, ActivationType.ReLU)
                .AddLayer(1, ActivationType.None);

            network.Randomize();
            ctx = network.CreateContext(18);
            ctx.SetLearningRate(minLearningRate, maxLearningRate);
        }

        private void DrawGraph()
        {
            this.Canvas.Children.Clear();
            GraphPoints.AddRange(new[]
            {
                DrawLine(0, 0, 0, 1, thickness: 3.0d),
                DrawLine(0, 0, 1, 0, thickness: 3.0d),
                DrawLine(0, 0, 0, -1, stroke: Brushes.DarkGray, thickness: 3.0d),
                DrawLine(0, 0, -1, 0, stroke: Brushes.DarkGray, thickness: 3.0d),
            });
        }


        private void DrawExpected()
        {
            for (double i = -1; i <= 1d; i += 0.001d)
            {
                ExpectedPoints.Add(
                    DrawPoint(new Point(i, Fit(i)), 4.0d, brush: Brushes.Green));
            }
        }

        private static double Fit(double i)
        {
            return Math.Sin(i * Math.PI) / Math.PI;
        }

        //private static double Fit(double i)
        //{
        //    if (Math.Floor(i * 4) % 2 == 0) return 0.25d;
        //    return -0.25d;
        //}


        private List<Point> predictions = new List<Point>();
        private DateTime startLearningTime;

        private async Task PredictAndDraw()
        {
            predictions.Clear();
            lock (network)
            {
                var input = new[] { 0d };
                for (double i = -1; i <= 1d; i += 0.001d)
                {
                    input[0] = i;
                    ctx.SetInput(input);
                    var val = network.Process(ctx)[0];
                    predictions.Add(new Point(i, val));
                }
            }

            await Dispatcher.InvokeAsync(() =>
            {
                var halfSize = 2d;
                if (PredictedPoints.Any())
                {
                    foreach (var p in predictions.Zip(PredictedPoints))
                    {
                        var scaled = ScaleUp(p.First);
                        Canvas.SetLeft(p.Second, scaled.X - halfSize);
                        Canvas.SetTop(p.Second, scaled.Y - halfSize);
                    }
                }
                else
                {
                    foreach (var p in predictions)
                    {
                        PredictedPoints.Add(
                            DrawPoint(new Point(p.X, p.Y), 4.0d, brush: Brushes.Red));
                    }
                }

                var duration = (DateTime.UtcNow - startLearningTime);
                var genPerTime = generations / duration.TotalSeconds;  

                StatusText.Text = $"G: {generations}  C: {cost:0.000000}  T: {duration:mm\\:ss\\.ff}   GpT: {genPerTime}";
            });

        }

        private async Task Learn()
        {
            startLearningTime = DateTime.UtcNow;
            var testTime = TimeSpan.FromMilliseconds(100);
            ctx.TrainingData.Clear();
            for (double i = -1; i <= 1d; i += 0.004d)
            {
                ctx.TrainingData.Add((new[] { i }, new[] { Fit(i) }, new double[1]));
            }

            var sw = new Stopwatch();
            while (isLearning)
            {
                sw.Start();
                int epoc = 0;
                while (sw.Elapsed < testTime && isLearning)
                {
                    cost = await network.Train(ctx);
                    generations++;
                    epoc++;
                }
                sw.Stop();
                sw.Reset();
                await PredictAndDraw();
            }
        }

        private UIElement DrawLine(double x1, double y1, double x2, double y2, Brush? stroke = null, double thickness = 1.0d)
        {
            if (stroke is null) stroke = Brushes.Black;

            var p1 = ScaleUp(new Point { X = x1, Y = y1 });
            var p2 = ScaleUp(new Point { X = x2, Y = y2 });

            var line = new Line()
            {
                X1 = p1.X,
                Y1 = p1.Y,
                X2 = p2.X,
                Y2 = p2.Y,
                StrokeThickness = thickness,
                SnapsToDevicePixels = true,
                Stroke = stroke,
            };
            Canvas.Children.Add(line);
            return line;
        }

        private UIElement DrawPoint(Point point, double size, Brush? brush = null, double thickness = 1.0d)
        {
            if (brush is null) brush = Brushes.Black;

            var p = ScaleUp(point);
            var halfSize = size / 2d;
            var e = new Ellipse()
            {
                Height = size,
                Width = size,
                Fill = brush,
            };
            Canvas.Children.Add(e);
            Canvas.SetLeft(e, p.X - halfSize);
            Canvas.SetTop(e, p.Y - halfSize);

            return e;
        }

        /// <summary>
        /// Scales points from pixels to graph
        /// </summary>
        private Point ScaleDown(Point p)
        {
            return new Point()
            {
                X = (p.X / ((Canvas.ActualWidth * graphSize))) - (graphSize / 2d),
                Y = ((p.Y / (Canvas.ActualHeight * graphSize)) - (graphSize / 2d)) * -1,
            };
        }

        /// <summary>
        /// Scales points from graph to pixels
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        private Point ScaleUp(Point p)
        {
            return new Point()
            {
                X = (p.X + (graphSize / 2d)) * (Canvas.ActualWidth / graphSize),
                Y = (((p.Y + (graphSize / 2d)) * (Canvas.ActualHeight / graphSize)) - Canvas.ActualHeight) * -1,
            };
        }

        private void LearnButton_Click(object sender, RoutedEventArgs e)
        {
            if (isLearning)
            {
                isLearning = false;
            }
            else
            {
                isLearning = true;
                Debug.WriteLine($"Learn Button: {System.Threading.Thread.CurrentThread.ManagedThreadId}");
                Task.Run(Learn);
            }
        }

        private void InitButton_Click(object sender, RoutedEventArgs e)
        {
            lock (network)
            {
                network.Randomize();
                generations = 0;
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "(Neural Network)|*.nn",
                DefaultExt = ".nn",
                Title = "Save Network"
            };

            if (dialog.ShowDialog() == true)
            {
                using var file = System.IO.File.OpenWrite(dialog.FileName);
                network.Save(file);
            }
        }

        private async void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog()
            {
                Title = "Open Network",
                Filter = "(Neural Network)|*.nn",
            };

            if (dialog.ShowDialog() == true)
            {
                using var file = System.IO.File.OpenRead(dialog.FileName);
                network = Network.Load(file);
                ctx = network.CreateContext(18);
                ctx.SetLearningRate(minLearningRate, maxLearningRate);

                await PredictAndDraw();
            }
        }
    }
}