
package jsat.classifiers;

/**
 *
 * @author Edward Raff
 */
public class CategoricalData
{
    private int n;//Number of different categories
    private String[] catNames;

    /**
     * 
     * @param n the number of categories
     */
    public CategoricalData(int n)
    {
        this.n = n;
        catNames = null;
    }

    /**
     * 
     * @return the number of possible categories there are for this category
     */
    public int getNumOfCategories()
    {
        return n;
    }
    
    public boolean isValidCategory(int i)
    {
        if (i < 0 || i >= n)
            return false;
        
        return true;
    }
    
    public String catName(int i)
    {
        if(catNames != null)
            return catNames[i];
        else
            return Integer.toString(i);
    }
    
    public CategoricalData copy()
    {
        CategoricalData copy = new CategoricalData(n);
        
        if(this.catNames != null)
        {
            String[] newCatNames = new String[n];
            for(int i = 0; i < n; i++)
                newCatNames[i] = new String(catNames[i]);
            copy.catNames = newCatNames;
        }
        
        return copy;
    }
    
    public static CategoricalData[] copyOf(CategoricalData[] orig)
    {
        CategoricalData[] copy = new CategoricalData[orig.length];
        for(int i = 0; i < copy.length; i++)
            copy[i] = orig[i].copy();
        return copy;
    }
    
}
