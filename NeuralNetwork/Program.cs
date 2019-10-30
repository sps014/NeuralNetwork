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
        static PolynomialRegression<double> lr = new PolynomialRegression<double>(2);

        static void Main()
        {
            Console.ForegroundColor= ConsoleColor.Yellow;
            read();
        }
        static void read()
        {
            List<double> input = new List<double>();
            List<double> output = new List<double>();
            //lr.OnTraining += (sender, e) => { 
               // string Text = e.Loss.ToString() + "  ->> ";
               // Console.WriteLine(Text);
            //};
            for (int i = 0; i < 90; i++)
            {
                input.Add(i / 100.0f);
                output.Add(input[i]*input[i]);

              
            }

            for (int i = 0; i < 1000; i++)
            {
                lr.Train(input.ToArray(), output.ToArray(),
                    new PolynomialRegression<double>.RegressionConfig()
                    {
                        Optimizer = PolynomialRegression<double>.OPTIMIZER.STOCHASTIC_GRADIENT_DESCENT,
                        Epochs = 10,
                        Shuffle = true
                    }) ;
            }

            MessageBox.Show("p(4)="+lr.Predict(0.6).ToString()+"\r\nanswer should be =0.36");

        }

    }
}
