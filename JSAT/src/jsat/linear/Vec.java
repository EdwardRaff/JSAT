
package jsat.linear;

/**
 *
 * @author Edward Raff
 */
public interface Vec
{
    public int length();

    public double get(int index);

    public void set(int index, double val);
    
    public Vec add(double c);
    public Vec add(Vec b);
    public Vec subtract(double c);
    public Vec subtract(Vec b);
    public Vec pairwiseMultiply(Vec b);
    public Vec multiply(double c);
    public Vec pairwiseDivide(Vec b);
    public Vec divide(double c);
    
    public void mutableAdd(double c);
    public void mutableAdd(Vec b);
    public void mutableSubtract(double c);
    public void mutableSubtract(Vec b);
    public void mutablePairwiseMultiply(Vec b);
    public void mutableMultiply(double c);
    public void mutablePairwiseDivide(Vec b);
    public void mutableDivide(double c);

    public Vec sortedCopy();

    public double min();
    public double max();
    public double sum();
    public double mean();
    public double standardDeviation();
    public double variance();
    public double median();
    public double skewness();
    public double kurtosis();
    
    public Vec copy();
    public Vec normalized();
    public void normalize();
    
    
    /**
     * Returns the p-norm distance between this and another vector y. 
     * @param p the distance type. 2 is the common value
     * @param y the other vector to compare against
     * @return the p-norm distance
     */
    public double pNormDist(double p, Vec y);
    
    public double pNorm(double p);
    
    /**
     * 
     * @param v the other vector
     * @return  the dot product of this vector and another
     */
    public double dot(Vec v);

    @Override
    public String toString();
    
    @Override
    public boolean equals(Object obj);

}
