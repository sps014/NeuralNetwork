using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Sonic.ML.Regression
{
    public class PolynomialRegression<T>
    {
        public double LearningRate { get; set; } = 0.0001;

        public List<double> Weights { get; private set; }

        //Increasing order of exponent ie last index of list has more power
        public uint Degree { get; private set; } = 1;

        public PolynomialRegression(uint degree=1)
        {
            Degree = degree;
            Weights = new List<double>();
            InitWeights();
        }
        private void InitWeights()
        {
            for (int i = 0; i <= Degree; i++)
            {
                Weights.Add(new Random(i).NextDouble());
            }
        }
        public double Predict(T input)
        {
            double val = 0;
            Parallel.For(0, Weights.Count, (i) =>
             {
                 val += System.Math.Pow((dynamic)input, i) * Weights[i];
             });

            return val;
        }
        public void Train(T[] inputData, T[] knownOutputs,RegressionConfig config=null)
        {
            if(config!=null)
            {
                for(int i=0;i<config.Epochs;i++)
                {
                    if(config.Shuffle)
                    {
                        Shuffle(inputData,knownOutputs);
                    }

                    if (config.Optimizer == OPTIMIZER.STOCHASTIC_GRADIENT_DESCENT)
                        SGD(inputData, knownOutputs);
                    else
                        BatchGD(inputData, knownOutputs);

                }
            }
            else
            {
                config = new RegressionConfig();
                if (config.Optimizer == OPTIMIZER.STOCHASTIC_GRADIENT_DESCENT)
                    SGD(inputData, knownOutputs);
                else
                    BatchGD(inputData, knownOutputs);
            }
           
        }
        void SGD(T[] inputData, T[] knownOutputs)
        {
            Parallel.For(0, inputData.Length,(i)=>
            {
                Parallel.For(0, Weights.Count, (j) =>
                {
                    double delWT = ((dynamic)knownOutputs[i] - Predict(inputData[i])) * System.Math.Pow((double)(dynamic)inputData[i], j);
                    Weights[j]+=delWT*LearningRate;
                });
                
                if (OnTraining != null)
                {
                    TrainingResponse res = new TrainingResponse();
                    res.Loss = 0.5f * System.Math.Pow((dynamic)knownOutputs[i] - Predict(inputData[i]), 2);
                    OnTraining?.Invoke(this, res);
                }
            });
        }
        void BatchGD(T[] inputData, T[] knownOutputs)
        {
            List<double> wts = new List<double>(new double[Weights.Count]);

            Parallel.For(0, inputData.Length, (i) =>
             {
                 Parallel.For(0, Weights.Count, (j) =>
                 {
                     double delWT = ((dynamic)knownOutputs[i] - Predict(inputData[i])) * System.Math.Pow((double)(dynamic)inputData[i], j);
                     wts[j] += delWT * LearningRate;
                 });
                 
                 if (OnTraining != null)
                 {
                     TrainingResponse res = new TrainingResponse();
                     res.Loss = 0.5f * System.Math.Pow((dynamic)knownOutputs[i] - Predict(inputData[i]), 2);
                     OnTraining?.Invoke(this, res);
                 }

             });

            Parallel.For(0, Weights.Count, (j) =>
            {
                Weights[j] += wts[j]*LearningRate;
            });

            //Slope = Slope + delta_m * LearningRate;
            //Bias = Bias + delta_c * LearningRate;
        }

        public class RegressionConfig
        {
            public int Epochs { get; set; } = 1;
            public OPTIMIZER Optimizer { get; set; } = OPTIMIZER.STOCHASTIC_GRADIENT_DESCENT;
            public bool Shuffle { get; set; } = false;
        }
        private void Shuffle(T[] Xs,T[] Ys)
        {
            Parallel.For(0, Xs.Length, (i) =>
            {
                int index = new Random(i).Next(0, Xs.Length);
                dynamic t1 = Xs[index];
                Xs[index] = Xs[i];
                Xs[i] = t1;

                dynamic t2 = Ys[index];
                Ys[index] = Ys[i];
                Ys[i] = t2;
            });
        }

        public enum OPTIMIZER
        {
            STOCHASTIC_GRADIENT_DESCENT,
            BATCH_GRADIENT_DESCENT
        }

        public delegate void OnTrainingHandler(object sender, TrainingResponse e);
        public event OnTrainingHandler OnTraining;
        public class TrainingResponse
        {
            public double Loss { get; set; }
        }

    }
}
