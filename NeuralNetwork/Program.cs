using Sonic.ML.Regression;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
    static class Program
    {
        static LinearRegression<double> lr = new LinearRegression<double>();

        static void Main()
        {
            Console.ForegroundColor= ConsoleColor.Yellow;
            read();
        }
        static void read()
        {
            List<double> input = new List<double>();
            List<double> output = new List<double>();
            lr.OnTraining += (sender, e) => { 
                string Text = e.Loss.ToString() + "  ->> ";
            };
            for (int i = 0; i < 90; i++)
            {
                input.Add(i / 100.0f);
                output.Add((i / 100f) * 100+2);
            }
            int a = 5;
            for (int i = 0; i < 2000; i++)
            {
                lr.Train(input.ToArray(), output.ToArray(),
                    new LinearRegression<double>.RegressionConfig()
                    {
                        Optimizer = LinearRegression<double>.OPTIMIZER.BATCH_GRADIENT_DESCENT,
                        Epochs = 10,
                        Shuffle = true
                    }) ;
            }

            MessageBox.Show(lr.Predict(0.99).ToString());

        }

    }
}
