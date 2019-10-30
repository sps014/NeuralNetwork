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
        static LogisticRegression<double> lr = new LogisticRegression<double>();

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
                if (i > 50)
                    output.Add(0);
                else
                    output.Add(1);
            }
            int a = 5;
            for (int i = 0; i < 3000; i++)
            {
                lr.Train(input.ToArray(), output.ToArray(),
                    new LogisticRegression<double>.RegressionConfig()
                    {
                        Optimizer = LogisticRegression<double>.OPTIMIZER.STOCHASTIC_GRADIENT_DESCENT,
                        Epochs = 10,
                        Shuffle = true
                    }) ;
            }

            MessageBox.Show("p(0.99)="+lr.Predict(0.1).ToString()+"\r\nm="+lr.Slope+"\r\nb="+lr.Bias);

        }

    }
}
