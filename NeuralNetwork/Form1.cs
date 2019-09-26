using Sonic.Math;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
   
    public partial class Form1 : Form
    {
        Sonic.AI.NeuralNetwork network;
        bool stop = false;

        public Form1()
        {
            InitializeComponent();
        }


        private  async void Thrad2()
        {
            await Task.Run(() =>
            {
                network = new Sonic.AI.NeuralNetwork(new uint[] { 3, 2, 2 });
                stop = false;
                List<xor> data = new List<xor>();
                data.Add(new xor(0, 1, 1));
                data.Add(new xor(0, 0, 0));
                data.Add(new xor(1, 1, 0));
                data.Add(new xor(1, 0, 1));


                Random r = new Random();
                const int MAX = 50000;
                for (int i = 0; i < MAX; i++)
                {

                    foreach (var d in data.OrderBy(a => Guid.NewGuid()).ToList())
                    {
                        Invoke((MethodInvoker)delegate ()
                        {
                            Text = i.ToString();

                            network.LearningRate = (float)(trackBar1.Value) / 10000.0f;
                        });
                        network.Train(d.inputs, d.target);
                    }

                    if (stop)
                    {
                        stop = false;
                        break;

                    }

                    Invoke((MethodInvoker)delegate ()
                    {
                        progressBar1.Maximum = MAX;
                        progressBar1.Value = i;
                        button2.Text = network.Layers[0].Weights[0, 0].ToString();
                        button3.Text = network.Layers[0].Weights[0, 1].ToString();
                        button4.Text = network.Layers[0].Weights[1, 0].ToString();
                        button5.Text = network.Layers[0].Weights[0, 1].ToString();
                        button6.Text = network.Layers[1].Weights[0, 0].ToString();

                        button8.Text = network.Layers[0].NeuronValue[0, 0].ToString();
                        button9.Text = network.Layers[0].NeuronValue[0, 1].ToString();
                        button10.Text = network.Layers[1].NeuronValue[0, 0].ToString();
                        button11.Text = network.Layers[1].NeuronValue[0, 1].ToString();
                        button12.Text = network.Layers[2].NeuronValue[0, 0].ToString();

                    });

                }


            });
        }
        class xor
        {
            public double[] target=new double[1];
            public double[] inputs=new double[3];
            public xor(double t,double i,double i1)
            {
                target[0] = t;
                inputs[0] = i;
                inputs[2] = 1;
                inputs[1] = i1;
            }
        }


        private void Button1_Click(object sender, EventArgs e)
        {
            if(!stop)
            Thrad2();
        }

        private void Button7_Click(object sender, EventArgs e)
        {
            network.Predict(new double[] { (double)numericUpDown1.Value, (double)numericUpDown2.Value ,1}).Print();
            button2.Text = network.Layers[0].Weights[0, 0].ToString();
            button3.Text = network.Layers[0].Weights[0, 1].ToString();
            button4.Text = network.Layers[0].Weights[1, 0].ToString();
            button5.Text = network.Layers[0].Weights[0, 1].ToString();
            button6.Text = network.Layers[1].Weights[0, 0].ToString();

            button8.Text = network.Layers[0].NeuronValue[0, 0].ToString();
            button9.Text = network.Layers[0].NeuronValue[1, 0].ToString();
            button10.Text = network.Layers[1].NeuronValue[0, 0].ToString();
            button11.Text = network.Layers[1].NeuronValue[1, 0].ToString();
            button12.Text = network.Layers[2].NeuronValue[0, 0].ToString();
        }

        private void Button13_Click(object sender, EventArgs e)
        {
            stop = true;
        }
    }
}
