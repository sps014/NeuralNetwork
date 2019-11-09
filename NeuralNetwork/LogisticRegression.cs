using System;
using System.Threading.Tasks;

namespace Sonic.ML.Regression
{
    public class LogisticRegression<T>
    {
        public double LearningRate { get; set; } = 0.0001;
        public double Slope { get; private set; } =new Random().NextDouble();
        public double Bias { get; private set; } = new Random(67).NextDouble();

        public double Predict(T input)
        {
            return Sigmoid(Slope * (dynamic)input + Bias);
        }
        private double Sigmoid(T input)
        {
            return 1 / (1.0 + System.Math.Exp(-(double)(dynamic)input));
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
        private double DSigmoid(T data)
        {
            return Predict(data) * (1.0-Predict(data));
        }
        void SGD(T[] inputData, T[] knownOutputs)
        {
            Parallel.For(0, inputData.Length,(i)=>
            {
                double delta_m= inputData[i] * ((dynamic)knownOutputs[i] - Predict(inputData[i]))*DSigmoid((inputData[i]));
                double delta_c = ((dynamic)knownOutputs[i] - Predict(inputData[i]));
                Slope += delta_m * LearningRate;
                Bias += delta_c * LearningRate;
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
            double delta_m = 0;
            double delta_c = 0;
            Parallel.For(0, inputData.Length, (i) =>
             {
                 delta_m += inputData[i] * ((dynamic)knownOutputs[i] - Predict(inputData[i])) * DSigmoid(inputData[i]);
                 delta_c += ((dynamic)knownOutputs[i] - Predict(inputData[i]));

                 if (OnTraining != null)
                 {
                     TrainingResponse res = new TrainingResponse();
                     res.Loss = 0.5f * System.Math.Pow((dynamic)knownOutputs[i] - Predict(inputData[i]), 2);
                     OnTraining?.Invoke(this, res);
                 }

             });


            Slope = Slope + delta_m * LearningRate;
            Bias = Bias + delta_c * LearningRate;
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
