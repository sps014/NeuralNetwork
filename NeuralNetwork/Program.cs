using Sonic.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace mltest
{
    static class Program
    {
        static NeuralNetwork network = new NeuralNetwork(new uint []{ 1, 2, 1 });
        static void Main()
        {
            Console.ForegroundColor= ConsoleColor.Yellow;
            read();
            Console.ReadKey();
        }
        static void read()
        {
            List<double> input = new List<double>();
            List<double> output = new List<double>();
            //lr.OnTraining += (sender, e) => { 
            // string Text = e.Loss.ToString() + "  ->> ";
            // Console.WriteLine(Text);
            //};

            for (int j = 0; j < 1000; j++)
            {
                for (int i = 1; i <= 100; i++)
                {
                    input.Add(i / 100.0f);
                    if (i < 50)
                        output.Add(0);
                    else
                        output.Add(1);
                    network.Train(new double[] { input.Last() }, new double[] { output.Last() });
                    Console.WriteLine(network.Predict(new double[] { 0.045 })[0,0]);

                }
            }

            network.Predict(new double[] { 0.045}).Print();
        }

    }
}
