using Sonic.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sonic.AI
{
    public class NeuralNetwork
    {
        public float LearningRate { get; set; } = 0.000001f;
        public List<Layer> Layers { get; set; }

        public NeuralNetwork(uint[] Dimensions)
        {
            List<uint> Dim = new List<uint>();
            Dim.AddRange(Dimensions);
            CreateLayers(Dim);
        }
        public NeuralNetwork(List<uint> Dimensions)
        {
            CreateLayers(Dimensions);
        }


        private void CreateLayers(List<uint> Dimensions)
        {
            Layers = new List<Layer>();

            for (int i = 0; i <Dimensions.Count; i++)
            {
                Layer prevLayer = i == 0 ? null : Layers[i - 1];

                Layer layer = new Layer(Dimensions[i], prevLayer);

                if (i == 0)
                {
                    layer.Type = LayerType.Input;
                    layer.Weights.Columns = Dimensions[i];
                    layer.Weights.Rows = Dimensions[i+1];

                }
                else if (i < Dimensions.Count - 1)
                {
                    layer.Type = LayerType.Intermediate;
                    layer.Weights.Columns = Dimensions[i];
                    layer.Weights.Rows = Dimensions[i+1];
                }
                else
                {
                    layer.Type = LayerType.Output;
                }

                //Random wts b/w 0-1
                layer.Weights.RandomizeMatrixValue();

                Layers.Add(layer);
            }
        }

        public Matrix<double> Predict(double[] inputs)
        {
            List<double> dls = new List<double>();
            dls.AddRange(inputs);
            return Predict(dls);
        }
        public Matrix<double> Predict(List<double> inputs)
        {
            if (Layers[0].Size != inputs.Count)
                throw new Exception("Inputs dimension incorrect not matching with neuron size.");

            Layers[0].NeuronValue.FromArray(inputs.ToArray());

            return SolveForward().Map(Sigmoid);
        }

        private Matrix<double> SolveForward()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].NeuronValue =Layers[i - 1].Weights* Layers[i - 1].NeuronValue;
            }

            return Layers[Layers.Count - 1].NeuronValue;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + System.Math.Exp(-x));
        }
        private double Dsigmoid(double x)
        {
            return x * (1.0 - x);
        }

        public void Train(double[] inputs,double[] answersArray)
        {
            var output = Predict(inputs);

            var answers = new Math.Matrix<double>();
            answers=answers.FromArray(answersArray);

            //Error
            var error = answers - output;

            //hidden error

            var gradient=Matrix<double>.Map(output,Dsigmoid);
            gradient.HadamardProduct(error);
            gradient *= LearningRate;

            var hidden_wt_delta = Matrix<double>.Multiply(gradient, Layers[Layers.Count - 2].NeuronValue);


            Layers[Layers.Count - 2].Weights += hidden_wt_delta.Transpose();

            //Layers[Layers.Count-2].Weights.Print();

            var prevError = error;
            for(int i=Layers.Count-2;i>0;i--)
            {
                var hidden_error = Matrix<double>.Multiply(Layers[i].Weights, prevError);
                prevError = hidden_error;
                gradient = Matrix<double>.Map(Layers[i].NeuronValue, Dsigmoid).Transpose();
                gradient.HadamardProduct(hidden_error);
                gradient *= LearningRate;

                hidden_wt_delta = Matrix<double>.Multiply(gradient, Layers[i - 1].NeuronValue);
                hidden_wt_delta.Transpose();
                Layers[i - 1].Weights += hidden_wt_delta;
            }

        }

    }
}
