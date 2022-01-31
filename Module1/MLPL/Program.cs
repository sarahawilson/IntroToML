using MLPL.Helper;

namespace MLPL
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Machine Learning Pipeline (MLPL)");
            DataManager myHelper = new DataManager();
            myHelper.PrintSomething();
            myHelper.LoadDataSet();
            myHelper.PrintDataSet();
        }
    }
}