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
        static MutatingNetwork net = new MutatingNetwork(new uint[] {1,2,1 });
        static LogisticRegression<double> lr = new LogisticRegression<double>();
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
                input.Add(i/100.0);
                output.Add(((i/100.0)>0.50)?0:1);
            }

            //lr.OnTraining += (sender, e) => { 
            // string Text = e.Loss.ToString() + "  ->> ";
            // Console.WriteLine(Text);
            //};

            for(int i=0;i<10000;i++)
            {
                lr.Train(input.ToArray(), output.ToArray(),new LogisticRegression<double>.RegressionConfig() { Shuffle=true });
                
            }
         

            MessageBox.Show(lr.Predict(0.6).ToString());
        }

    }
}
