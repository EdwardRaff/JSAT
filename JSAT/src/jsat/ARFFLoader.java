
package jsat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;

/**
 *
 * @author Edward Raff
 */
public class ARFFLoader
{
    public static List<DataPoint> loadArffFile(File file) 
    {
        ArrayList<DataPoint> list = new ArrayList<DataPoint>();
        
        BufferedReader br;
        
        try
        {
            br = new BufferedReader(new FileReader(file));
        }
        catch (FileNotFoundException ex)
        {
            //No File
            return list;
        }
        
        int numOfVars = 0;
        int numReal = 0;
        List<Boolean> isReal = new ArrayList<Boolean>();
        List<HashMap<String, Integer>> catVals = new  ArrayList<HashMap<String, Integer>>();
        String line = null;
        CategoricalData[] categoricalData = null;
        try
        {
            boolean atData = false;
            while( (line = br.readLine()) != null )
            {
                if(line.startsWith("%") || line.trim().isEmpty())
                    continue;///Its a comment, skip
                
                line = line.trim();
                
                if(line.startsWith("@") && !atData)
                {
                    line = line.substring(1).toLowerCase();
                    
                    
                    if(line.startsWith("data"))
                    {
                        categoricalData = new CategoricalData[numOfVars-numReal];
                        
                        int k = 0;
                        for(int i = 0; i < catVals.size(); i++)
                        {
                            if(catVals.get(i) != null)
                                categoricalData[k++] = new CategoricalData(catVals.get(i).size());
                        }
                        
                        atData = true;
                        continue;
                    }
                    else if(!line.startsWith("attribute"))
                        continue;
                    numOfVars++;
                    line = line.substring("attribute".length()).trim();//Remove the space, it could be multiple spaces
                    
                    line = line.replace("\t", " ");
                    if(line.startsWith("'"))
                        line = line.replaceFirst("'.+?'", "placeHolder");
                    String[] tmp = line.split("\\s+", 2);
                    
                    if(tmp[1].trim().equals("real") || tmp[1].trim().startsWith("integer"))
                    {
                        numReal++;
                        isReal.add(true);
                        catVals.add(null);
                    }
                    else//Not correct, but we arent supporting anything other than real and categorical right now
                    {
                        isReal.add(false);
                        String cats = tmp[1].replace("{", "").replace("}", "").trim();
                        if(cats.endsWith(","))
                            cats = cats.substring(0, cats.length()-1);
                        String[] catValsRaw =  cats.split(",");
                        HashMap<String, Integer> tempMap = new HashMap<String, Integer>();
                        for(int i = 0; i < catValsRaw.length; i++)
                            tempMap.put(catValsRaw[i].trim(), i);
                        catVals.add(tempMap);
                    }
                }
                else if(atData && !line.isEmpty())
                {
                    if(line.contains("?"))//We dont handle missing data
                        continue;
                    String[] tmp = line.split(",");
                    
                    DenseVector vec = new DenseVector(numReal);
                    
                    int[] cats = new int[numOfVars - numReal];
                    int k = 0;//Keeping track of position in cats
                    for(int i  = 0; i < tmp.length; i++)
                    {
                        if(isReal.get(i))
                            vec.set(i - k, Double.parseDouble(tmp[i].trim()));
                        else//Categorical
                        {
                            cats[k++] = catVals.get(i).get(tmp[i].trim().toLowerCase());
                        }
                    }
                    
                    list.add(new DataPoint(vec, cats, categoricalData)); 
                }
            }
        }
        catch (IOException ex)
        {
            
        }
        
        return list;
    }
}
