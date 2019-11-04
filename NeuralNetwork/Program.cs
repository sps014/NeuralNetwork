using Sonic.ML;
using Sonic.ML.Regression;
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
        static NeuralNetwork net = new NeuralNetwork(new int[] { 1, 2, 1 }, new string[] { "sigmoid", "sigmoid", "sigmoid" });
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

            for (int i = 0; i < 100; i++)
            {
                input.Add(i/100f);
                output.Add(i<50?0:1);
            }

            //lr.OnTraining += (sender, e) => { 
            // string Text = e.Loss.ToString() + "  ->> ";
            // Console.WriteLine(Text);
            //};

            for(int i=0;i<10000;i++)
            {
                for (int j = 0; j < input.Count; j++)
                {
                    net.BackPropagate(new float[] { (float)input[j] }, new float[] { (float)output[j] });
                }
            }
         

            Console.WriteLine(net.FeedForward(new float[] { 1 })[0]);
        }

    }
}
