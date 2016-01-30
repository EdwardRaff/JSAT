
package jsat.classifiers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Edward Raff
 */
public class CategoricalData implements Cloneable, Serializable
{

    private static final long serialVersionUID = 5783467611963064930L;
    private int n;//Number of different categories
    private List<String> catNames;
    private String categoryName;

    /**
     * 
     * @param n the number of categories
     */
    public CategoricalData(int n)
    {
        this.n = n;
        catNames = new ArrayList<String>(n);
        for(int i = 0; i < n; i++)
            catNames.add("Option " + (i+1));
        categoryName = "No Name";
    }

    /**
     * 
     * @return the number of possible categories there are for this category
     */
    public int getNumOfCategories()
    {
        return n;
    }
    
    /**
     * Returns true if the given input is a valid category index for this object. Missing values (negative categories) do not count. 
     * @param i the index for a category in this object
     * @return {@code true} if it was a valid category, {@code false} otherwise. 
     */
    public boolean isValidCategory(int i)
    {
        if (i < 0 || i >= n)
            return false;
        
        return true;
    }
    
    public String getOptionName(int i)
    {
        if(i < 0)
            return "Missing Value";
        else if(catNames != null)
            return catNames.get(i);
        else
            return Integer.toString(i);
    }

    public String getCategoryName()
    {
        return categoryName;
    }

    public void setCategoryName(String categoryName)
    {
        this.categoryName = categoryName;
    }
    
    /**
     * Sets the name of one of the value options. Duplicate names are not allowed. 
     * Trying to set the name of a non existent option will result in false being 
     * returned. 
     * <br>
     * All names will be converted to lower case
     * 
     * @param name the name to give
     * @param i  the ith index to set. 
     * @return true if the name was set. False if the name could not be set. 
     */
    public boolean setOptionName(String name, int i)
    {
        name = name.toLowerCase();
        if(i < 0 || i >= n)
            return false;
        else if(catNames.contains(name))
            return false;
        catNames.set(i, name);
        
        return true;
    }
    public CategoricalData clone()
    {
        CategoricalData copy = new CategoricalData(n);
        
        if(this.catNames != null)
            copy.catNames = new ArrayList<String>(this.catNames);
        
        return copy;
    }
    
    public static CategoricalData[] copyOf(CategoricalData[] orig)
    {
        CategoricalData[] copy = new CategoricalData[orig.length];
        for(int i = 0; i < copy.length; i++)
            copy[i] = orig[i].clone();
        return copy;
    }
    
}
