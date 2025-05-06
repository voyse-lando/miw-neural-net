namespace miw_neural_net
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            var outputFunc = (string msg) => {
                output1.AppendText(msg);
                output1.AppendText(Environment.NewLine);
            };
            NeuralNetwork.Output = outputFunc;
            XOR.Output = outputFunc;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            XOR.Start();
        }
    }
}
