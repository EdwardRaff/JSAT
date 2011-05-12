
package jsat.classifiers;

/**
 *
 * @author Edward Raff
 */
public class CategoricalData
{
    private int n;//Number of different categories

    /**
     * 
     * @param n the number of categories
     */
    public CategoricalData(int n)
    {
        this.n = n;
    }

    /**
     * 
     * @return the number of possible categories there are for this category
     */
    public int getN()
    {
        return n;
    }
    
    public boolean isValidCategory(int i)
    {
        if (i < 0 || i >= n)
            return false;
        
        return true;
    }
    
    
}
