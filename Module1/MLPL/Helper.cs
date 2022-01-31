using System.Data;

namespace MLPL.Helper
{
    public class DataManager
    {
        DataTable thisData = new DataTable();


        public void LoadDataSet() 
        {
            Console.WriteLine("Loading in ML Data Set");
            //TODO: Implement 
            thisData.Columns.Add("Name");
            thisData.Columns.Add("Email");

            DataRow curRow = thisData.NewRow();
            curRow["Name"] = "Sarah";
            curRow["Email"] = "Sarah@gmail.com";

            thisData.Rows.Add(curRow);
        }

        public void PrintDataSet()
        {
            //TODO: Implement
            foreach (DataRow row in thisData.Rows) 
            {
                string Name = row["Name"].ToString();
                Console.WriteLine(Name);
                string Email = row["Email"].ToString();
                Console.WriteLine(Email);
            }
        }

        public void PrintSomething()
        {
            Console.WriteLine("Something");
        }
    }
}
