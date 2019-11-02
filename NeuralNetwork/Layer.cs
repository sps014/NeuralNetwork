using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sonic.ML
{
    public class Layer
    {
        public Layer PrevLayer { get; set; }
        public LayerType Type { get; set; } = LayerType.Intermediate;
        public uint Size { get; set; }
        public Math.Matrix<double> Weights { get; set; } = new Math.Matrix<double>();

        public Math.Matrix<double> NeuronValue { get; set; } = new Math.Matrix<double>();

        public Layer(uint size, Layer prevLayer=null)
        {
            Size = size;
            NeuronValue = new Math.Matrix<double>(size,size);
            PrevLayer = prevLayer;
        }


    }
    public enum LayerType
    {
        Input,
        Intermediate,
        Output
    }
}
