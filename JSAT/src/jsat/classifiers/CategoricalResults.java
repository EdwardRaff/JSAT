
package jsat.classifiers;

import java.util.Arrays;

/**
 * This class represents the probabilities for each possible result classification. 
 * @author Edward Raff
 */
public class CategoricalResults implements Cloneable
{
    private int n;//The number of categories
    private double[] probabilities;

    /**
     * Create a new Categorical Results, values will default to all zero. 
     * @param numCategories the number of options to support. 
     */
    public CategoricalResults(int numCategories)
    {
        n = numCategories;
        probabilities = new double[numCategories];
    }
    
    /**
     * Returns the number of classes that are in the result. 
     * @return the class count 
     */
    public int size()
    {
        return probabilities.length;
    }

    /**
     * Sets the probability that a sample belongs to a given category. 
     * @param cat the category
     * @param prob the value to set, may be greater then one. 
     * @throws IndexOutOfBoundsException if a non existent category is specified
     * @throws ArithmeticException if the value set is negative or not a number
     */
    public void setProb(int cat, double prob)
    {
        if(cat > probabilities.length)
            throw new IndexOutOfBoundsException("There are only " + probabilities.length + " posibilties, " + cat + " is invalid");
        else if(prob < 0 || Double.isInfinite(prob) || Double.isNaN(prob))
            throw new ArithmeticException("Only zero and positive values are valid, not " + prob);
        probabilities[cat] = prob;
    }
    
    /**
     * Increments the stored probability that a sample belongs to a given category 
     * @param cat the category
     * @param prob  the value to increment by, may be greater then one. 
     * @throws IndexOutOfBoundsException if a non existent category is specified
     * @throws ArithmeticException if the value set is negative or not a number
     */
    public void incProb(int cat, double prob)
    {
        if(cat > probabilities.length)
            throw new IndexOutOfBoundsException("There are only " + probabilities.length + " posibilties, " + cat + " is invalid");
        else if(prob < 0 || Double.isInfinite(prob) || Double.isNaN(prob))
            throw new ArithmeticException("Only zero and positive values are valid, not " + prob);
        probabilities[cat] += prob;
    }
    
    /**
     * Returns the category that is the most likely according to the current probability values 
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
    
    /**
     * Adjusts the probabilities by dividing each value by the total sum, so 
     * that all values are in the range [0, 1]
     */
    public void normalize()
    {
        double sum = 0;
        for(double d : probabilities)
            sum += d;
        if(sum != 0)
            divideConst(sum);
    }
    
    /**
     * Returns the stored probability for the given category
     * @param cat the category
     * @return the associated probability
     */
    public double getProb(int cat)
    {
        return probabilities[cat];
    }
    
    /**
     * Creates a deep clone of this 
     * @return a deep clone
     */
    @Override
    public CategoricalResults clone()
    {
        CategoricalResults copy = new CategoricalResults(n);
        copy.probabilities = Arrays.copyOf(probabilities, probabilities.length);
        return copy;
    }
}
