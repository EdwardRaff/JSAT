
package jsat.classifiers;

/**
 *
 * @author Edward Raff
 */
public class CategoricalResults
{
    private int n;//The number of categories
    private double[] probabilities;

    public CategoricalResults(int numCategories)
    {
        n = numCategories;
        probabilities = new double[numCategories];
    }

    /**
     * Sets the probability that a sample belongs to a given category. 
     * @param cat
     * @param prob 
     */
    public void setProb(int cat, double prob)
    {
        probabilities[cat] = prob;
    }
    
    /**
     * 
     * @return the the most likely category
     */
    public int mostLikely()
    {
        int top = 0;
        for(int i = 1; i < probabilities.length; i++)
        {
            if(probabilities[i] > probabilities[top])
                top = i;
        }
        
        return top;
    }
    
    /**
     * Divides all the probabilities by a constant value in order to scale them
     * @param c the constant to divide all probabilities by
     */
    public void divideConst(double c)
    {
        for(int i = 0; i < probabilities.length; i++)
            probabilities[i]/=c;
    }
    
    public double getProb(int cat)
    {
        return probabilities[cat];
    }
    
    
}
