using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Sonic.Math
{
    public class Matrix<T>
    {
        private static Random random=new Random(DateTime.Now.Millisecond);

        private List<List<T>> data;
        public T this[int i,int j]
        {
            get
            {
                if (i > rows || j > columns)
                    throw new Exception($"Matrix Index out of bound [{i},{j}] for size [{rows},{columns}]");

                return data[i][j];
            }
            set
            {

                if (i > rows || j > columns)
                    throw new Exception($"Matrix Index out of bound [{i},{j}] for size [{rows},{columns}]");



                DataChanged?.Invoke(this, new DataChangeArgs()
                {
                    NewValue = value,
                    OldValue = data[i][j],
                    IndexCol = (uint)j,
                    IndexRow = (uint)i
                }) ;

                data[i][j] = value;
            }
        }

        private uint rows;
        private uint columns;

        public uint Rows
        {
            get
            {
                return rows;
            }
            set
            {
                RowChanged?.Invoke(this, rows);
                rows = value;
                RecreateMatrix();
            }
        }
        public uint Columns
        {
            get
            {
                return columns;
            }
            set
            {
                ColumnChanged?.Invoke(this, columns);
                columns = value;
                RecreateMatrix();
            }
        }

        public Matrix()
        {
            data = new List<List<T>>();
        }

        public Matrix(int rows, int columns)
        {
            data = new List<List<T>>();
            Rows = (uint)rows;
            Columns = (uint)columns;
        }
        public Matrix(uint rows,uint columns)
        {
            data = new List<List<T>>();
            Rows = rows;
            Columns = columns;

        }

        public void RandomizeMatrixValue()
        {
            RecreateMatrixWithRandom();
        }


        private void RecreateMatrix()
        {
            data.Clear();
            for(int i=0;i<rows;i++)
            {
                data.Add(new List<T>());
                for (int j = 0; j < columns; j++)
                {
                    data[i].Add(default(T));
                }
            }
        }
        private void RecreateMatrixWithRandom()
        {
            data.Clear();
            for (int i = 0; i < rows; i++)
            {
                data.Add(new List<T>());
                for (int j = 0; j < columns; j++)
                {
                    double a=random.NextDouble();
                    data[i].Add((T)(dynamic)a);
                }
            }
        }


        public Matrix<T> Add(Matrix<T> B)
        {
            if (B.Columns != Columns || B.Rows!= Rows)
                throw new Exception("For Addition dimension of both matrix should be same -> " + $"A:{Rows}x{Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(Rows, Columns);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    dynamic a = this[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a + b;
                    this[(int)i, (int)j] = C[(int)i, (int)j];
                });
            });

            return C;
        }
        public static Matrix<T> Add(Matrix<T> A,Matrix<T> B)
        {
            if (B.Columns != A.Columns || B.Rows != A.Rows)
                throw new Exception("For Addition dimension of both matrix should be same -> "+$"A:{A.Rows}x{A.Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(A.Rows, B.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    dynamic a = A[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a + b;
                });
            });

            return C;
        }
        public static Matrix<T> operator +(Matrix<T> A,Matrix<T> B)
        {
            return Matrix<T>.Add(A, B);
        }



        public Matrix<T> Subtract(Matrix<T> B)
        {
            if (B.Columns != Columns || B.Rows != Rows)
                throw new Exception("For Subtraction dimension of both matrix should be same -> " + $"A:{Rows}x{Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(Rows, Columns);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    dynamic a = this[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a - b;
                    this[(int)i, (int)j] = C[(int)i, (int)j];
                });
            });

            return C;
        }
        public static Matrix<T> Subtract(Matrix<T> A, Matrix<T> B)
        {
            if (B.Columns != A.Columns || B.Rows != A.Rows)
                throw new Exception("For Subtraction dimension of both matrix should be same -> " + $"A:{A.Rows}x{A.Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(A.Rows, B.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    dynamic a = A[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a - b;
                });
            });

            return C;
        }
        public static Matrix<T> operator -(Matrix<T> A, Matrix<T> B)
        {
            return Matrix<T>.Subtract(A, B);
        }


        public Matrix<T> Multiply(Matrix<T> B)
        {
            if (Columns != B.Rows)
                throw new Exception("Can't Multiply with unequal number of columns and rows of the first and second matrix respectively ->" + $"A:{Rows}x{Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(Rows, B.Columns);

            Parallel.For(0, Rows, (i) =>
            {
                for(int j=0;j<B.Columns;j++)
                {
                    for (int k = 0; k < Columns; k++)
                    {
                        dynamic a = this[(int)i, k];
                        dynamic b = B[k, j];
                        C[(int)i, j] +=  a*b;
                    }
                }
            });

            C.CopyTo(this);

            return C;
        }
        public static Matrix<T> Multiply(Matrix<T> A, Matrix<T> B)
        {
            if (A.Columns != B.Rows)
                throw new Exception("Can't Multiply with unequal number of columns and rows of the first and second matrix respectively -> " + $"A:{A.Rows}x{A.Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(A.Rows, B.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                for (int j = 0; j < B.Columns; j++)
                {
                    for (int k = 0; k < A.Columns; k++)
                    {
                        dynamic a = A[(int)i, k];
                        dynamic b = B[k, j];
                        C[(int)i, j] += a * b;
                    }
                }
            });

            return C;
        }
        public static Matrix<T> operator *(Matrix<T> A, Matrix<T> B)
        {
            return Matrix<T>.Multiply(A, B);
        }


        public Matrix<T> Multiply(T B)
        {
            Matrix<T> C = new Matrix<T>(Rows, Columns);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                 {
                     C[(int)i, (int)j] = (dynamic)this[(int)i, (int)j] * (dynamic)B;
                     this[(int)i, (int)j] = C[(int)i, (int)j];
                 });
            });

            return C;
        }
        public static Matrix<T> Multiply(Matrix<T> A,T B)
        {
            Matrix<T> C = new Matrix<T>(A.Rows, A.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    C[(int)i, (int)j] = (dynamic)A[(int)i, (int)j] * (dynamic)B;
                });
            });

            return C;
        }
        public static Matrix<T> operator *(Matrix<T> A, T B)
        {
            return Matrix<T>.Multiply(A, B);
        }


        public Matrix<T> HadamardProduct(Matrix<T> B)
        {
            if (B.Columns != Columns || B.Rows != Rows)
                throw new Exception("For Hadmard Product dimension of both matrix should be same -> "+$"A:{Rows}x{Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(Rows, B.Columns);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    dynamic a = this[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a * b;
                    this[(int)i, (int)j] = C[(int)i, (int)j];
                });
            });

            return C;
        }
        public static Matrix<T> HadamardProduct(Matrix<T> A,Matrix<T> B)
        {
            if (B.Columns != A.Columns || B.Rows != A.Rows)
                throw new Exception("For Hadmard Product dimension of both matrix should be same -> " + $"A:{A.Rows}x{A.Columns} and B:{B.Rows}x{B.Columns}");

            Matrix<T> C = new Matrix<T>(A.Rows, B.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    dynamic a = A[(int)i, (int)j];
                    dynamic b = B[(int)i, (int)j];
                    C[(int)i, (int)j] = a * b;
                });
            });

            return C;
        }

        public Matrix<T> Transpose()
        {
            Matrix<T> C = new Matrix<T>(Columns, Rows);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    C[(int)j, (int)i] = this[(int)i, (int)j]; 
                });
            });

            C.CopyTo(this);

            return C;
        }
        public static Matrix<T> Transpose(Matrix<T> A)
        {
            Matrix<T> C = new Matrix<T>(A.Columns, A.Rows);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    C[(int)j, (int)i] = A[(int)i, (int)j];
                });
            });

            return C;
        }


        public static Matrix<T> Identity(int n)
        {
            Matrix<T> identity = new Matrix<T>(n, n);
            for (int i = 0; i < n; ++i)
                identity[i,i] = (dynamic)1;

            return identity;
        }


        public T GetAt(int i,int j)
        {
            return this[i,j];
        }
        public void SetAt(int i, int j,T Value)
        {
            this[i, j]=Value;
        }

        public Matrix<T> Map(Func<T,T> func)
        {

            Matrix<T> C = new Matrix<T>(Rows, Columns);

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    dynamic a = this[(int)i, (int)j];
                    C[(int)i, (int)j] = func(a);
                    this[(int)i, (int)j] = C[(int)i, (int)j];
                });
            });

            return C;
        }
        public static Matrix<T> Map(Matrix<T> A,Func<T, T> func)
        {

            Matrix<T> C = new Matrix<T>(A.Rows, A.Columns);

            Parallel.For(0, A.Rows, (i) =>
            {
                Parallel.For(0, A.Columns, (j) =>
                {
                    dynamic a = A[(int)i, (int)j];
                    C[(int)i, (int)j] = func(a);
                });
            });

            return C;
        }


        public Matrix<T> FromArray(T[] array,bool rowVector=true)
        {
            Matrix<T> c;

            if(rowVector)
            {
                c = new Matrix<T>(array.Length,1);
                Parallel.For(0, array.Length, (i) =>
                  {
                      c[i, 0] = array[i];
                  });
            }
            else
            {
                c = new Matrix<T>(1,array.Length);
                Parallel.For(0, array.Length, (i) =>
                {
                    c[0, i] = array[i];
                });
            }

            c.CopyTo(this);

            return c;
        }
        public void CopyTo(Matrix<T> Target)
        {
            Target.Rows = Rows;
            Target.Columns = Columns;

            Parallel.For(0, Rows, (i) =>
            {
                Parallel.For(0, Columns, (j) =>
                {
                    Target[(int)i, (int)j] = this[(int)i, (int)j];
                });
            });
        }
        public  string ToString(int dashMultipiler=12)
        {
            StringBuilder builder = new StringBuilder("");

            for(int i=0;i<Rows;i++)
            {
                if(i==0)
                {
                    for (int k = 0; k < dashMultipiler * Columns; k++)
                        builder.Append("-");
                    builder.Append("\r\n");

                }

                for (int j=0;j<Columns;j++)
                {
                    if (j == 0)
                        builder.Append("|");

                    string value = data[i][j].ToString();
                    builder.Append(value.PadLeft(7)+" ".PadLeft(System.Math.Abs(7-value.Length))+"|");

                }
                builder.Append("\r\n");


                for (int k = 0; k < dashMultipiler * Columns; k++)
                    builder.Append("-");
                builder.Append("\r\n");

            }

            return builder.ToString();
        }
        public void Print(int dash=12)
        {
            System.Windows.Forms.MessageBox.Show(ToString(dash));
        }


        public delegate void RowChangedHandler(Matrix<T> matrix, uint oldValue);
        public event RowChangedHandler RowChanged;

        public delegate void ColumnChangedHandler(Matrix<T> matrix, uint oldValue);
        public event ColumnChangedHandler ColumnChanged;

        public delegate void DataChangedHandler(Matrix<T> matrix, DataChangeArgs e);
        public event DataChangedHandler DataChanged;

        public class DataChangeArgs:EventArgs
        {
            public uint IndexRow { get; set; }
            public uint IndexCol { get; set; }
            public T OldValue { get; set; }
            public T NewValue { get; set; }
        }
    }
}
